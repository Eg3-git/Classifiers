import rospy
import geometry_msgs.msg
import pandas as pd
import time

from tf2_msgs.msg import TFMessage
from omni_msgs.msg import OmniButtonEvent
from sensor_msgs.msg import JointState
import csv
import os

button_stat = OmniButtonEvent()
button_stat.white_button = 0
button_stat.grey_button = 0


def callBack(msg):
    global filename
    #  haptic pose end effector

    # rospy.loginfo(msg.pose.position.x)
    # rospy.loginfo(msg.pose.position.y)
    # rospy.loginfo(msg.pose.position.z)
    _time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    content = [_time, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w, button_stat.white_button, button_stat.grey_button]
    csv_writeline(filename, content)

def callBack1(msg):
    global filename2, button_stat
    # print(time.time())

    # ur3e end effector

    # rospy.loginfo(msg.transforms[0].transform.translation.x)
    # rospy.loginfo(msg.transforms[0].transform.translation.y)
    # rospy.loginfo(msg.transforms[0].transform.translation.z)
    for transform in msg.transforms:
        if transform.child_frame_id == "tool0_controller" and transform.header.frame_id == "base":
            _time = transform.header.stamp.secs + transform.header.stamp.nsecs * 1e-9
            content = [_time, transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            # content = [time.time(), transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w, button_stat.white_button, button_stat.grey_button]
            csv_writeline(filename1, content)

def callBack2(msg):
    global filename3
    # for girpper joints gripper_joint_states

        # print("joints recorded")
    _time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    content = [_time, msg.position[0], msg.velocity[0]]
    csv_writeline(filename3, content)

def callBack3(msg):

    global button_stat
    button_stat.white_button = msg.white_button
    button_stat.grey_button = msg.grey_button

def callBack4(msg):
    global filename2
    #ur3e joint_state
    # rospy.loginfo(msg.pose.position.x)
    # rospy.loginfo(msg.pose.position.y)
    # rospy.loginfo(msg.pose.position.z)
    if msg.position[0] == 0.0:
        print("pass")
    elif len(msg.position) >= 6:
        # print("joints recorded")
        _time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        content = [_time, msg.position[0], msg.position[1], msg.position[2], msg.position[3], msg.position[4], msg.position[5]]
        csv_writeline(filename2, content)



def record():
    print("runnning")
    rospy.init_node('haptic_data_aquisition')
    rospy.Subscriber('/phantom_right/phantom/pose',geometry_msgs.msg.PoseStamped, callBack, queue_size=1)
    rospy.Subscriber('/tf', TFMessage, callBack1)
    rospy.Subscriber('/gripper/joint_states', JointState, callBack2)
    rospy.Subscriber('/phantom_right/phantom/button', OmniButtonEvent, callBack3)
    rospy.Subscriber('/joint_states', JointState, callBack4)
    
    
    rospy.spin()
    print("ended")



def csv_writeline(filename, filecontent):
    with open(filename, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(filecontent)




if __name__ == '__main__' :
    name = ""
    name = input("Name the file: ")
    filename = name + "_haptics_end_effector_pose.csv"
    filename1 = name + "_ur3e_end_effectors_pose.csv"
    filename2 = name + "_ur3e_joint_states.csv"
    filename3 = name + "_gripper_joint_states.csv"
    header = ["time", "x", "y", "z", "qx", "qy", "qz", "qw", "white_button_state", "grey_button_state"]
    header1 = ["time", "x", "y", "z", "qx", "qy", "qz", "qw", "white_button_state", "grey_button_state"]
    header2 = ["time", "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    header3 = ["time", "gripper_joint_position", "gripper_joint_velocity"]
   
    try: 
        open(filename, 'x')
        open(filename1, 'x')
        open(filename2, 'x')
        open(filename3, 'x')
        
        csv_writeline(filename, header)
        csv_writeline(filename1, header1)
        csv_writeline(filename2, header2)
        csv_writeline(filename3, header3)
        record()
    except:
        if os.stat(filename).st_size == 0:
            l = 0
        else:
            file = pd.read_csv(filename)
            l = len(file)
        print("file already exists with",l,"lines")
        rp = input("Continue? yes/no\n")
        if (rp == 'yes'):
            if(l == 0):
                csv_writeline(filename, header)
                csv_writeline(filename1, header1)
                csv_writeline(filename2, header2)
                csv_writeline(filename3, header3)
            record()
    
