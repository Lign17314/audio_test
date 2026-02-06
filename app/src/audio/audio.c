
/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: sample/cvi_sample_audio.c
 * Description:example for audio api flow
 * such as audio in, audio out, audio trancode flow
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <sys/time.h>
#include <stdbool.h>
#include "cvi_type.h"
#include "cvi_sys.h"
#include <cvi_common.h>
#include "cvi_buffer.h"
#include "cvi_comm_vb.h"
#include "cvi_comm_isp.h"
#include "cvi_comm_3a.h"
#include "cvi_comm_sns.h"
#include <cvi_comm_vi.h>
#include <cvi_comm_vpss.h>
#include <cvi_comm_vo.h>
#include <cvi_comm_venc.h>
#include <cvi_comm_vdec.h>
#include <cvi_comm_region.h>
#include "cvi_comm_adec.h"
#include "cvi_comm_aenc.h"
#include "cvi_comm_aio.h"
#include "cvi_audio.h"
#include "cvi_defines.h"
#define ACODEC_ADC "/dev/cvitekaadc"
#define ACODEC_DAC "/dev/cvitekadac"

#define SMP_AUD_UNUSED_REF(X) ((X) = (X))
#define AUDIO_ADPCM_TYPE ADPCM_TYPE_DVI4 /* ADPCM_TYPE_IMA, ADPCM_TYPE_DVI4*/
#define G726_BPS MEDIA_G726_32K          /* MEDIA_G726_16K, MEDIA_G726_24K ... */

// static AAC_TYPE_E     gs_enAacType = AAC_TYPE_AACLC;
// static AAC_BPS_E     gs_enAacBps  = AAC_BPS_32K;
// static AAC_TRANS_TYPE_E gs_enAacTransType = AAC_TRANS_TYPE_ADTS;

#define FILE_NAME_LEN 128
#define AEC_LOOP_RECORD_RUN 1
#define AEC_LOOP_RECORD_STOP 0

/* WAV */
#define ID_RIFF 0x46464952
#define ID_WAVE 0x45564157
#define ID_FMT 0x20746d66
#define ID_DATA 0x61746164

struct riff_wave_header
{
    unsigned int riff_id;
    unsigned int riff_sz;
    unsigned int wave_id;
};

struct chunk_header
{
    unsigned int id;
    unsigned int sz;
};

struct chunk_fmt
{
    unsigned short audio_format;
    unsigned short num_channels;
    unsigned int sample_rate;
    unsigned int byte_rate;
    unsigned short block_align;
    unsigned short bits_per_sample;
};

typedef struct
{
    int AiDev;
    int AiChn;
    bool bVqe;
    int cap_loop;
    int ChnCnt;
    int record_status;
} ST_VQE_RECORD_TEST_STRUCT;

typedef struct
{
    int AoDev;
    int AoChn;
    int ChnCnt;
    unsigned int u32PtNumPerFrm;
    FILE *fp_playfile;
    int stop_flag;
} ST_VQE_PLAY_TEST_STRUCT;

typedef struct
{
    int sample_rate;
    int channel;
    int preiod_size;
    int codec;
    // PAYLOAD_TYPE_E eType;
    bool bVqeOn;
    char filename[FILE_NAME_LEN];
    int Chnsample_rate;
    int record_time;
} stAudPara;

static void dump_audiodata(char *filename, char *buf, unsigned int len)
{
    FILE *fp;

    if (filename == NULL)
    {
        return;
    }

    fp = fopen(filename, "ab+");
    fwrite(buf, 1, len, fp);
    fclose(fp);
}

int running = 1;
void sigint_handler_sample(int signal)
{
    SMP_AUD_UNUSED_REF(signal);
    running = 0;
}

void register_inthandler(void)
{
    signal(SIGINT, sigint_handler_sample);
    signal(SIGHUP, sigint_handler_sample);
    signal(SIGTERM, sigint_handler_sample);
}

static CVI_BOOL _update_aec_setting(AI_TALKVQE_CONFIG_S *pstAiVqeTalkAttr)
{
    if (pstAiVqeTalkAttr == NULL)
        return CVI_FALSE;

    AI_AEC_CONFIG_S default_AEC_Setting;

    memset(&default_AEC_Setting, 0, sizeof(AI_AEC_CONFIG_S));
    default_AEC_Setting.para_aec_filter_len = 13;
    default_AEC_Setting.para_aes_std_thrd = 37;
    default_AEC_Setting.para_aes_supp_coeff = 60;
    pstAiVqeTalkAttr->stAecCfg = default_AEC_Setting;
    pstAiVqeTalkAttr->u32OpenMask = LP_AEC_ENABLE | NLP_AES_ENABLE | NR_ENABLE | AGC_ENABLE;
    printf("pstAiVqeTalkAttr:u32OpenMask[0x%x]\n", pstAiVqeTalkAttr->u32OpenMask);
    return CVI_FALSE;
}

static CVI_BOOL _update_agc_anr_setting(AI_TALKVQE_CONFIG_S *pstAiVqeTalkAttr)
{
    if (pstAiVqeTalkAttr == NULL)
        return CVI_FALSE;

    pstAiVqeTalkAttr->u32OpenMask |= (NR_ENABLE | AGC_ENABLE | DCREMOVER_ENABLE);

    AUDIO_AGC_CONFIG_S st_AGC_Setting;
    AUDIO_ANR_CONFIG_S st_ANR_Setting;

    st_AGC_Setting.para_agc_max_gain = 0;
    st_AGC_Setting.para_agc_target_high = 2;
    st_AGC_Setting.para_agc_target_low = 72;
    st_AGC_Setting.para_agc_vad_ena = CVI_TRUE;
    st_ANR_Setting.para_nr_snr_coeff = 15;
    st_ANR_Setting.para_nr_init_sile_time = 0;

    pstAiVqeTalkAttr->stAgcCfg = st_AGC_Setting;
    pstAiVqeTalkAttr->stAnrCfg = st_ANR_Setting;

    pstAiVqeTalkAttr->para_notch_freq = 0;
    printf("pstAiVqeTalkAttr:u32OpenMask[0x%x]\n", pstAiVqeTalkAttr->u32OpenMask);
    return CVI_TRUE;
}

static PAYLOAD_TYPE_E _sample_audio_get_codec(int s32Opt)
{
    PAYLOAD_TYPE_E eType = PT_G726;

    if (s32Opt == 0)
    {
        // printf("[cvi_info] Codec G726\n");
        eType = PT_G726;
    }
    else if (s32Opt == 1)
    {
        printf("[cvi_info] Codec G711A\n");
        eType = PT_G711A;
    }
    else if (s32Opt == 2)
    {
        printf("[cvi_info] Codec G711Mu\n");
        eType = PT_G711U;
    }
    else if (s32Opt == 3)
    {
        printf("[cvi_info] Codec PT_ADPCMA\n");
        eType = PT_ADPCMA;
    }
    else if (s32Opt == 4)
    {
        printf("[cvi_info] Codec AAC_LC\n");
        eType = PT_AAC;
    }
    else if (s32Opt == 5)
    {
        printf("[cvi_info] Codec SBC\n");
        eType = PT_SBC;
    }
    else
    {
        printf("[cvi_info] Enter invalid num ....select g726\n");
        eType = PT_G726;
    }

    return eType;
}

CVI_S32 SAMPLE_AUDIO_RECORD_PCM_FORMAT_FILE(void *argv)
{
    stAudPara param = {
        .sample_rate = 16000,
        .channel = 1,
        .preiod_size = 1024,
        .codec = 1,
        .bVqeOn = 0,
        .filename = "out.pcm",
        .Chnsample_rate = 16000,
        .record_time = 20,
    };

    int AiDev = 0; /*only support 0 dev */
    int AiChn = 0;
    int s32Ret = 0;
    bool ReSam_flag = false;
    struct timespec end;
    struct timespec now;
    int AudMaxChn = 3;
    AI_TALKVQE_CONFIG_S stAiVqeTalkAttr;
    AI_TALKVQE_CONFIG_S *pstAiVqeTalkAttr = (AI_TALKVQE_CONFIG_S *)&stAiVqeTalkAttr;
    stAudPara *pstAudioparam = (stAudPara *)&param;

    if (!pstAudioparam)
    {
        printf("[fatal error] fpAenc is NULL,fuc:%s,line:%d\n", __func__, __LINE__);
        return -1;
    }

    int sample_rate = pstAudioparam->sample_rate;
    unsigned int Chnsample_rate = pstAudioparam->Chnsample_rate;
    int channel = pstAudioparam->channel;
    int u32PtNumPerFrm = pstAudioparam->preiod_size;
    bool bVqe = pstAudioparam->bVqeOn;
    int record_time = pstAudioparam->record_time;
    PAYLOAD_TYPE_E enType = _sample_audio_get_codec(pstAudioparam->codec); /*PT_G711A,PT_G711U,PT_G726 */

    register_inthandler();
    FILE *fpAi = fopen(pstAudioparam->filename, "ab+");

    if (!fpAi)
    {
        printf("fpAi open fail\n");
        return -1;
    }

    // STEP 1:set ai and vqe attr
    AIO_ATTR_S AudinAttr;

    AudinAttr.enSamplerate = (AUDIO_SAMPLE_RATE_E)sample_rate;
    AudinAttr.u32ChnCnt = AudMaxChn;
    AudinAttr.enSoundmode = (channel == 2 ? AUDIO_SOUND_MODE_STEREO : AUDIO_SOUND_MODE_MONO);
    AudinAttr.enBitwidth = AUDIO_BIT_WIDTH_16;
    AudinAttr.enWorkmode = AIO_MODE_I2S_MASTER;
    AudinAttr.u32EXFlag = 0;
    AudinAttr.u32FrmNum = 10;                  /* only use in bind mode */
    AudinAttr.u32PtNumPerFrm = u32PtNumPerFrm; /* sample_rate/fps */
    AudinAttr.u32ClkSel = 0;
    AudinAttr.enI2sType = AIO_I2STYPE_INNERCODEC;

    /*if you want to use vqe ,ai chn must 2chn.*/
    /*if you don't need to use vqe, you can skip this step*/
    if (bVqe)
    {
        memset(&stAiVqeTalkAttr, 0, sizeof(AI_TALKVQE_CONFIG_S));
        if (((AudinAttr.enSamplerate == AUDIO_SAMPLE_RATE_8000) ||
             (AudinAttr.enSamplerate == AUDIO_SAMPLE_RATE_16000)) &&
            channel == 2)
        {

            pstAiVqeTalkAttr->s32WorkSampleRate = AudinAttr.enSamplerate;
            _update_agc_anr_setting(pstAiVqeTalkAttr);
            _update_aec_setting(pstAiVqeTalkAttr);
        }
        else
        {
            printf("[error] AEC will need to setup record in to channel Count = 2\n");
            printf("[error] VQE only support on 8k/16k sample rate. current[%d]\n", AudinAttr.enSamplerate);
        }
    }

    s32Ret = CVI_AUDIO_INIT();
    if (s32Ret != CVI_SUCCESS)
    {
        printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
        goto ERROR3;
    }

    // STEP 2:start ai and aenc
    s32Ret = CVI_AI_SetPubAttr(AiDev, &AudinAttr);
    if (s32Ret != CVI_SUCCESS)
    {
        printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
        goto ERROR3;
    }
    s32Ret = CVI_AI_Enable(AiDev);
    if (s32Ret != CVI_SUCCESS)
    {
        printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
        goto ERROR3;
    }

    s32Ret = CVI_AI_EnableChn(AiDev, AiChn);
    if (s32Ret != CVI_SUCCESS)
    {
        printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
        goto ERROR2;
    }

    if (bVqe == true)
    {
        s32Ret = CVI_AI_SetTalkVqeAttr(AiDev, AiChn, 0, 0, &stAiVqeTalkAttr);
        if (s32Ret != CVI_SUCCESS)
        {
            printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
            goto ERROR1;
        }

        s32Ret = CVI_AI_EnableVqe(AiDev, AiChn);
        if (s32Ret != CVI_SUCCESS)
        {
            printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
            goto ERROR1;
        }
    }

    // if ((enType == PT_G711A || enType == PT_G711U) && (Chnsample_rate != 8000))
    // {
    //     Chnsample_rate = 8000;
    //     printf("G711 only support sr 8000,change to 8000.\n");
    // }

    if ((Chnsample_rate != (unsigned int)AudinAttr.enSamplerate))
    {
        s32Ret = CVI_AI_EnableReSmp(AiDev, AiChn, Chnsample_rate);
        if (s32Ret != CVI_SUCCESS)
        {
            printf("[error],[%s],[line:%d],\n", __func__, __LINE__);
            goto ERROR1;
        }
        ReSam_flag = true;
    }

    AUDIO_FRAME_S stFrame;
    AEC_FRAME_S stAecFrm;

    clock_gettime(CLOCK_MONOTONIC, &now);
    end.tv_sec = now.tv_sec + record_time;
    end.tv_nsec = now.tv_nsec;

    int s32OutputChnCnt = channel;

    while (running)
    {
        if (record_time)
        {
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec > end.tv_sec || (now.tv_sec == end.tv_sec && now.tv_nsec >= end.tv_nsec))
                break;
        }

        s32Ret = CVI_AI_GetFrame(AiDev, AiChn, &stFrame, &stAecFrm, -1);
        if (s32Ret)
        {
            printf("[error] ai getframe error\n");
            break;
        }
        printf("%s,%d %d\r\n", __func__, __LINE__, stFrame.u32Len);
        fwrite(stFrame.u64VirAddr[0], 1, stFrame.u32Len * s32OutputChnCnt * 2, fpAi);
    }

ERROR1:
    if (bVqe == true)
        CVI_AI_DisableVqe(AiDev, AiChn);
    if (ReSam_flag == true)
        CVI_AI_DisableReSmp(AiDev, AiChn);
    CVI_AI_DisableChn(AiDev, AiChn);
ERROR2:
    CVI_AI_Disable(AiDev);
ERROR3:
    fclose(fpAi);
    CVI_AUDIO_DEINIT();

    return 0;
}

int main()
{
    CVI_BOOL MSG_INIT = CVI_FALSE;

    if (!MSG_INIT)
    {
        CVI_MSG_Init();
        sleep(1);
        MSG_INIT = CVI_TRUE;
    }
    SAMPLE_AUDIO_RECORD_PCM_FORMAT_FILE(NULL);
    if (MSG_INIT)
    {
        CVI_MSG_Deinit();
        MSG_INIT = CVI_FALSE;
    }
    return 0;
}