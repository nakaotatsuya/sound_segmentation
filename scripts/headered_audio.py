#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from audio_common_msgs.msg import AudioData
from sound_segmentation.msg import AudioHeaderData

class HeaderedAudio():
    def __init__(self):
        rospy.Subscriber("~audio", AudioData, self.audio_cb, queue_size=1000, buff_size=2**24)
        self.pub = rospy.Publisher("~output", AudioHeaderData, queue_size=1)

    def audio_cb(self, msg):
        pub_msg = AudioHeaderData()
        pub_msg.data = msg.data
        pub_msg.header.stamp = rospy.Time.now()
        self.pub.publish(pub_msg)

if __name__ == "__main__":
    rospy.init_node("headered_audio")
    a = HeaderedAudio()
    rospy.spin()
        
