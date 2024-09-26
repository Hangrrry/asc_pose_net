from pymavlink import mavutil

# 连接到 PX4（默认情况下仿真环境使用 UDP 端口 14550）
connection = mavutil.mavlink_connection('udp:localhost:14445')

# 等待并接收飞控的心跳信号，确认连接成功
if(connection.wait_heartbeat()):
	print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

while True:
    # 接收消息
    msg = connection.recv_match(type='STATUSTEXT', blocking=True)
    if msg:
        print("Received message:", msg.text)