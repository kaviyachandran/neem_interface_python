# neem_interface_python

Provides a Python wrapper to interact with NEEMs.

All NEEM-related functionality is provided by the `NEEMInterface` class (in `src/neem_interface_python/neem_interface.py`).
A general wrapper to use rosprolog from Python is provided by the `Prolog` class (in `src/neem_interface_python/rosprolog_client.py`)

`NEEMInterface` and the `Prolog` client talk to rosprolog via [roslibpy](https://roslibpy.readthedocs.io/en/latest/), which in turn needs a running [rosbridge_server](http://wiki.ros.org/rosbridge_server).
When using `NEEMInterface` or `Prolog` from other ROS packages, make sure to start the rosbridge server in your launch file:

```xml
<include file="$(find rosbridge_server)/launch/rosbridge_websocket.launch"/>
```

Alternatively, you can `include` the launch file provided by this package:

```xml
<include file="$(find neem_interface_python)/launch/rosbridge.launch"/>
```

Additionally, `NEEMInterface` and `Prolog` need to read the `ROS_MASTER_URI` to be able to communicate with ROS. Make sure that you source your ROS workspace (`source catkin_ws/devel/setup.bash`) in any processes which use `NEEMInterface` or `Prolog`.