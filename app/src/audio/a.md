
x 8000000 3C7000
connect 0 EAGLE_EYES test123456
ifconfig vnet0 hw ether 88:00:33:77:12:57
ifconfig vnet0 192.168.31.68
route add default gw 192.168.31.1
ifconfig vnet0 up


tftp -p -l out.pcm -r out.pcm 192.168.31.59