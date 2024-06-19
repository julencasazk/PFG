from opcua import Server
from opcua import ua

# Create a server instance
server = Server()


ip = "10.172.7.140"


# Create a server instance
server = Server()

# Define the server endpoint
url = f"opc.tcp://{ip}:4840"
server.set_endpoint(url)

# Create a new address space
address_space = server.register_namespace("ConveyorControlSystem")
print(f"Address space created at {url} with Namespace index {address_space}")


root = server.get_root_node()

# Server Variables
start=root.add_variable(address_space, "Start", False, varianttype=ua.VariantType.Boolean)
start.set_writable(True)
batches=root.add_variable(address_space, "Completed_Batches", 0, varianttype=ua.VariantType.Int16)
batches.set_writable(True)
# Raspberry Pi Nodes
raspi = server.nodes.objects.add_object(address_space, "RaspberryPi")

# Raspberry Pi Variables
prox_1=raspi.add_variable(address_space, "Prox_Sensor_1", False, varianttype=ua.VariantType.Boolean)
prox_1.set_writable(True)
prox_2=raspi.add_variable(address_space, "Prox_Sensor_2", False, varianttype=ua.VariantType.Boolean)
prox_2.set_writable(True)
prox_3=raspi.add_variable(address_space, "Prox_Sensor_3", False, varianttype=ua.VariantType.Boolean)
prox_3.set_writable(True)
speed_1=raspi.add_variable(address_space, "Speed_1", False, varianttype=ua.VariantType.Boolean) 
speed_1.set_writable(True)
speed_2=raspi.add_variable(address_space, "Speed_2", False, varianttype=ua.VariantType.Boolean) 
speed_2.set_writable(True)
speed_3=raspi.add_variable(address_space, "Speed_3", False, varianttype=ua.VariantType.Boolean) 
speed_3.set_writable(True)
current_speed_1=raspi.add_variable(address_space, "CurrentSpeed_1", 2, varianttype=ua.VariantType.Int16) # Total speed number from 0 to 7
current_speed_1.set_writable(True)
current_speed_2=raspi.add_variable(address_space, "CurrentSpeed_2", 2, varianttype=ua.VariantType.Int16) # Total speed number from 0 to 7
current_speed_2.set_writable(True)

led_1=raspi.add_variable(address_space, "Led_1", False, varianttype=ua.VariantType.Boolean)
led_1.set_writable(True)
led_2=raspi.add_variable(address_space, "Led_2", False, varianttype=ua.VariantType.Boolean)
led_2.set_writable(True)

direction=raspi.add_variable(address_space, "Direction", False, varianttype=ua.VariantType.Boolean)
direction.set_writable(True)
running=raspi.add_variable(address_space, "Running", False, varianttype=ua.VariantType.Boolean)
running.set_writable(True)

state=raspi.add_variable(address_space, "State", 0, varianttype=ua.VariantType.Int16)
state.set_writable(True)

batch_finished=raspi.add_variable(address_space, "Batch_Finished", False, varianttype=ua.VariantType.Boolean)
batch_finished.set_writable(True)



# Raspberry Pi Methods
def update_batch_number():
    if batch_finished.get_value():
            barcode_data.set_value("")
            n = batches.get_value()
            batches.set_value(n+1)
            current_speed_2.set_value(2) 

            batch_finished.set_value(False)


# Jetson Nodes
jetson = server.nodes.objects.add_object(address_space, "Jetson")

# Jetson Variables

barcode_data=jetson.add_variable(address_space, "Barcode_Data", "", varianttype=ua.VariantType.String)
barcode_data.set_writable(True)
box_dimensions=jetson.add_variable(address_space, "Box_Dimensions", [0.0, 0.0, 0.0], varianttype=ua.VariantType.Float)
box_dimensions.set_writable(True)
frame_count=jetson.add_variable(address_space, "Frame_Count", 0, varianttype=ua.VariantType.Int16)
frame_count.set_writable(True)

activate_cam_1=jetson.add_variable(address_space, "Activate_Cam_1", False, varianttype=ua.VariantType.Boolean)
activate_cam_1.set_writable(True)
activate_cam_2=jetson.add_variable(address_space, "Activate_Cam_2", False, varianttype=ua.VariantType.Boolean)
activate_cam_2.set_writable(True)
activate_cam_3=jetson.add_variable(address_space, "Activate_Cam_3", False, varianttype=ua.VariantType.Boolean)
activate_cam_3.set_writable(True)
activate_depth=jetson.add_variable(address_space, "Activate_Depth", False, varianttype=ua.VariantType.Boolean)
activate_depth.set_writable(True)



# Jetson Methods
def check_barcode_detected(normal_speed):
    bardata = barcode_data.get_value()
    statenum = state.get_value()
    if bardata != "": # Barcode correctly read
        activate_cam_1.set_value(False)
        activate_cam_2.set_value(False)
        activate_cam_3.set_value(False)
        current_speed_1.set_value(normal_speed)
        current_speed_2.set_value(7)
    else:
        current_speed_1.set_value(normal_speed)
        current_speed_2.set_value(normal_speed)
        if statenum == 0:
            activate_cam_1.set_value(False)
            activate_cam_2.set_value(False)
            activate_cam_3.set_value(False)
        elif statenum == 1: # Box detected
            activate_cam_1.set_value(True) # Front camera
            activate_cam_3.set_value(True) # Upper camera
            activate_cam_2.set_value(False) # Back camera
        elif statenum == 4:
            activate_cam_1.set_value(False)
            activate_cam_3.set_value(False)
            activate_cam_2.set_value(True)

def check_dimensions():
    statenum = state.get_value()
    if statenum == 1:
        activate_depth.set_value(True)
    elif statenum == 5:
        activate_depth.set_value(False)

# Start the server
server.start()
print(f"Server started at {ip}")

# Run the server indefinitely
try:

    normal_speed = 2
    current_speed_1.set_value(normal_speed)
    current_speed_2.set_value(normal_speed)
    
    while True:
        update_batch_number()
        check_barcode_detected(normal_speed)
        check_dimensions()

except KeyboardInterrupt:
    # Stop the server on keyboard interrupt
    print(f"Completed batches since Server started: {batches.get_value()}")
    server.stop()






