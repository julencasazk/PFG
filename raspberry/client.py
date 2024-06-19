import RPi.GPIO as GPIO
import piplates.RELAYplate as RELAY
from State import State
import time
from opcua import Client

# GPIO Pin numbers
PROX1_GPIO = 16
PROX2_GPIO = 17
PROX3_GPIO = 4

# Relay Numbers
DIR = 1
SPEED_A = 4
SPEED_B = 3
SPEED_C = 2
INDEF = 5
LED_2 = 6 # Backside led
LED_1 = 7 # Frontside Led

def set_speed(speed):
    # Speed must be between 0 and 7
    outs = RELAY.relaySTATE(0)
    if speed > 7:
        print("Error: Speed must be between 0 and 7")
        return
    outs = outs & 0b1110001
    outs = outs | (speed << 1)
    RELAY.relayALL(0, outs)

states = ["WAITING", "BOX_DETECTED","BOX_FIRST_HALF", "BOX_CROSSING_MIDDLE","BOX_SECOND_HALF", "BOX_EXITING", "BOX_EXITED"]

def process_state(state, prox_1, prox_2, prox_3, speed_1, speed_2):
    if states[state] == "WAITING": #0
        set_speed(0)
        RELAY.relayOFF(0, LED_1) # LED1
        RELAY.relayOFF(0, LED_2) # LED2

        if prox_1:
            state += 1
            print(f"Changed stated to {state}")

    elif states[state] == "BOX_DETECTED": #1
        set_speed(speed_1)
        RELAY.relayON(0, LED_1)
        if not prox_1:
            state += 1
            print(f"Changed stated to {state}")

    elif states[state] == "BOX_FIRST_HALF": #2
        # RELAY.relayON(0, LED_1)
        set_speed(speed_1)

        if prox_2:
            state += 1
            print(f"Changed stated to {state}")

    elif states[state] == "BOX_CROSSING_MIDDLE": #3
        set_speed(speed_1) 

        if not prox_2:
            state += 1
            print(f"Changed stated to {state}")

    elif states[state] == "BOX_SECOND_HALF": #4
        RELAY.relayOFF(0, LED_1)
        RELAY.relayON(0, LED_2)
        set_speed(speed_2)

        if prox_3:
            state += 1
            print(f"Changed stated to {state}")

    elif states[state] == "BOX_EXITING": #5
        set_speed(speed_2)

        if not prox_3:
            state = 0
            print(f"Changed stated to {state}")

    else:
        state = 0
        print(f"Changed stated to {state}")
    return state

def read_relay_states():
    # Read the state of all relays and convert to boolean list
    relay_states = RELAY.relaySTATE(0)
    return [(relay_states & (1 << i)) != 0 for i in range(8)]


pin = -1
state = 0

if __name__ == "__main__":

    ip = "10.172.7.140"
    # OPC-UA Client setup
    client = Client(f"opc.tcp://{ip}:4840")
    try:
        client.connect()
        print("Connected to OPC-UA Server at IP: {ip}")

        # Server variables
        server_state = client.get_node("ns=2;i=16")
        server_speed_a = client.get_node("ns=2;i=7")
        server_speed_b = client.get_node("ns=2;i=8")
        server_speed_c = client.get_node("ns=2;i=9")
        server_led_1 = client.get_node("ns=2;i=11")
        server_led_2 = client.get_node("ns=2;i=12")
        server_direction = client.get_node("ns=2;i=13")
        server_running = client.get_node("ns=2;i=14")
        server_prox_1 = client.get_node("ns=2;i=4")
        server_prox_2 = client.get_node("ns=2;i=5")
        server_prox_3 = client.get_node("ns=2;i=6")
        server_start = client.get_node("ns=2;i=1")
        batch_finished = client.get_node("ns=2;i=17")
        current_speed_1 = client.get_node("ns=2;i=10")
        current_speed_2 = client.get_node("ns=2;i=11")

        # Direction
        RELAY.relayON(0, DIR)

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PROX1_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PROX2_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PROX3_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        speed = 2 # Change this to change speed
        try:
            while True:
                prox_1 = not GPIO.input(PROX1_GPIO)
                prox_2 = not GPIO.input(PROX2_GPIO)
                prox_3 = not GPIO.input(PROX3_GPIO)
                speed_1 = current_speed_1.get_value()
                speed_2 = current_speed_2.get_value()
                print(f"Speed1: {speed_1}, Speed2: {speed_2}")
                state = process_state(state, prox_1, prox_2, prox_3, speed_1, speed_2)
                if state == 5:
                    batch_finished.set_value(True)


                relay_values = read_relay_states()
                # Update all server variables related to conveyor control
                server_state.set_value(state)
                server_prox_1.set_value(prox_1)
                server_prox_2.set_value(prox_2)
                server_prox_3.set_value(prox_3)
                server_speed_a.set_value(relay_values[SPEED_A-1])
                server_speed_b.set_value(relay_values[SPEED_B-1])
                server_speed_c.set_value(relay_values[SPEED_C-1])
                server_led_1.set_value(relay_values[LED_1-1])
                server_led_2.set_value(relay_values[LED_2-1])
                server_direction.set_value(relay_values[DIR-1])



        except KeyboardInterrupt:
            # Por seguridad apagar todos los relés al acabar el programa
            RELAY.relayALL(0, 0)
    finally:
        client.disconnect()
        print("Disconnected from OPC-UA Server")
