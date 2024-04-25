import os
from urllib.parse import urlparse

import roslibpy

ros_host = urlparse(os.environ["ROS_MASTER_URI"]).hostname
assert ros_host is not None, "ros host is None"
rosbridge_port = 9090

ros_client = roslibpy.Ros(ros_host, rosbridge_port)
ros_client.run()
