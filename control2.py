import serial
import os
import asyncio
import time


def raise_arm(PWM):
    PWM.change_angle(5)

def drop_arm(PWM):
    PWM.change_angle(25)

def proccess_received(Ser,rec):
    if(rec == b'begin\r\n'):
        turn_page()
        send_serial('end')
    else:
        print('receive error')


async def read_serial():
    have_receive = 0
    last_size = 0
    ser = serial.Serial('/dev/ttyS1',115200)
    if ser.isOpen == False:
        ser.open()
    try:
        while True:
            size = ser.inWaiting()
            if size!=0:
                if have_receive == 0:
                    await asyncio.sleep(0.5)
                    have_receive = 1
                    last_size = size
                else :
                    if(size != last_size):
                        await asyncio.sleep(0.5)
                        last_size = size
                    else:
                        response=ser.read(size)
                        print(response)
                        proccess_received(ser,response)
                        ser.flushInput()
                        last_size = 0
                        have_receive = 0
                        await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        ser.close()  # 关闭串口

def send_serial(s):
    ser = serial.Serial('/dev/ttyS1',115200)
    if ser.isOpen == False:
        ser.open()
    ser.write(b'over\r\n')
#   os.system('echo ' + s + ' > /dev/ttyS1')

def slow(P,now,target):
    while(now != target):
        if(now > target):
            now = now - 1
        elif (now < target):
            now = now + 1
        P.change_angle(now)
        time.sleep(0.1)

def turn_page():
    print('begin turn page')
    P1 = pwm('1') #arm
    P0 = pwm('0') 
    P1.enable()
    P0.enable()
    yaw = 15
    pitch = 25
    P0.change_angle(15)
    P1.change_angle(25)
    time.sleep(3)
    slow(P0,15,18)
    slow(P1,25,15)
    slow(P1,15,7)
    time.sleep(0.7)
    slow(P1,5,15)
    slow(P1,15,25)
    slow(P0,20,15)
    slow(P0,15,5)
    time.sleep(0.5)
    slow(P0,10,15)
    time.sleep(3)
    print('finish turn page')



async def main():
    await asyncio.gather(read_serial())
    
class pwm:
    def __init__(self,index):
        self.index = index
#       print(os.path.join('/sys/class/pwm/pwmchip0','pwm'+index))
        if(os.path.exists(os.path.join('/sys/class/pwm/pwmchip' + index,'pwm0'))):
            print('device exists')
        else:
            os.system('echo 0 > /sys/class/pwm/pwmchip' + index + '/export')
            print('create pwm' + index + ' successfully')
        os.system('echo 50000000 > /sys/class/pwm/pwmchip' + index + '/pwm0/period')
        os.system('echo 47500000 > /sys/class/pwm/pwmchip' + index + '/pwm0/duty_cycle')

    def change_angle(self,num):
        if(num > 25):
            num = 25
        if(num < 5): 
            num = 5
        num = (500 - num)*100000
        os.system('echo ' + str(int(num)) + ' > /sys/class/pwm/pwmchip' + self.index + '/pwm0/duty_cycle')

    def enable(self):
        os.system('echo 1 > /sys/class/pwm/pwmchip' + self.index + '/pwm0/enable')

    def disable(self):
        os.system('echo 0 > /sys/class/pwm/pwmchip' + self.index + '/pwm0/enable')

    

       


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
