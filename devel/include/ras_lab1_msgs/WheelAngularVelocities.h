// Generated by gencpp from file ras_lab1_msgs/WheelAngularVelocities.msg
// DO NOT EDIT!


#ifndef RAS_LAB1_MSGS_MESSAGE_WHEELANGULARVELOCITIES_H
#define RAS_LAB1_MSGS_MESSAGE_WHEELANGULARVELOCITIES_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace ras_lab1_msgs
{
template <class ContainerAllocator>
struct WheelAngularVelocities_
{
  typedef WheelAngularVelocities_<ContainerAllocator> Type;

  WheelAngularVelocities_()
    : header()
    , W1(0.0)
    , W2(0.0)  {
    }
  WheelAngularVelocities_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , W1(0.0)
    , W2(0.0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef float _W1_type;
  _W1_type W1;

   typedef float _W2_type;
  _W2_type W2;




  typedef boost::shared_ptr< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> const> ConstPtr;

}; // struct WheelAngularVelocities_

typedef ::ras_lab1_msgs::WheelAngularVelocities_<std::allocator<void> > WheelAngularVelocities;

typedef boost::shared_ptr< ::ras_lab1_msgs::WheelAngularVelocities > WheelAngularVelocitiesPtr;
typedef boost::shared_ptr< ::ras_lab1_msgs::WheelAngularVelocities const> WheelAngularVelocitiesConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace ras_lab1_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'ras_lab1_msgs': ['/home/wasp/catkin_ws/src/ras_lab1/ras_lab1_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ae0c4a7348b1a8bfa79b856330d95cad";
  }

  static const char* value(const ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xae0c4a7348b1a8bfULL;
  static const uint64_t static_value2 = 0xa79b856330d95cadULL;
};

template<class ContainerAllocator>
struct DataType< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ras_lab1_msgs/WheelAngularVelocities";
  }

  static const char* value(const ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
\n\
float32 W1\n\
float32 W2\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
";
  }

  static const char* value(const ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.W1);
      stream.next(m.W2);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct WheelAngularVelocities_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ras_lab1_msgs::WheelAngularVelocities_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "W1: ";
    Printer<float>::stream(s, indent + "  ", v.W1);
    s << indent << "W2: ";
    Printer<float>::stream(s, indent + "  ", v.W2);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RAS_LAB1_MSGS_MESSAGE_WHEELANGULARVELOCITIES_H
