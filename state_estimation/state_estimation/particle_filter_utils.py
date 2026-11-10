import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf_transformations
# import tf2_ros
import time


def wrap_angle(angle):
    """Wrap an angle or array of angles to the interval [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(angle, reference):
    """Return the shortest signed angular difference from reference to angle."""
    return wrap_angle(angle - reference)


def _normalized_weights(weights):
    """Return finite, non-negative weights normalized to sum to one."""
    normalized = np.asarray(weights, dtype=float)
    if normalized.ndim != 1:
        raise ValueError('weights must be one-dimensional')
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError('weights must be finite and non-negative')

    total = np.sum(normalized)
    if total <= 0.0:
        raise ValueError('weights must have a positive sum')
    return normalized / total


def weighted_pose_mean(poses, weights):
    """Compute the weighted mean of planar [x, y, yaw] poses."""
    poses = np.asarray(poses, dtype=float)
    normalized = _normalized_weights(weights)
    if poses.ndim != 2 or poses.shape != (normalized.size, 3):
        raise ValueError('poses must have shape (len(weights), 3)')

    mean = np.empty(3, dtype=float)
    mean[:2] = np.sum(poses[:, :2] * normalized[:, None], axis=0)
    sin_sum = np.dot(np.sin(poses[:, 2]), normalized)
    cos_sum = np.dot(np.cos(poses[:, 2]), normalized)

    # A circular mean is undefined for a uniform or exactly opposed angular
    # distribution. Use an actual particle direction instead of an angle chosen
    # by floating-point cancellation in that case.
    if np.hypot(sin_sum, cos_sum) < 1.0e-12:
        mean[2] = wrap_angle(poses[np.argmax(normalized), 2])
    else:
        mean[2] = np.arctan2(sin_sum, cos_sum)
    return mean


def weighted_pose_covariance(poses, weights, mean=None):
    """Compute planar pose covariance using wrapped residuals for yaw."""
    poses = np.asarray(poses, dtype=float)
    normalized = _normalized_weights(weights)
    if poses.ndim != 2 or poses.shape != (normalized.size, 3):
        raise ValueError('poses must have shape (len(weights), 3)')
    if mean is None:
        mean = weighted_pose_mean(poses, normalized)
    else:
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (3,):
            raise ValueError('mean must have shape (3,)')

    residuals = poses - mean
    residuals[:, 2] = angle_difference(poses[:, 2], mean[2])
    return (residuals * normalized[:, None]).T @ residuals


def planar_covariance_to_ros(covariance):
    """Embed an [x, y, yaw] covariance in a ROS 6D pose covariance."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3):
        raise ValueError('planar covariance must have shape (3, 3)')

    ros_covariance = np.zeros((6, 6))
    planar_indices = np.array([0, 1, 5])
    ros_covariance[np.ix_(planar_indices, planar_indices)] = covariance
    return ros_covariance.flatten().tolist()


class CircularArray(object):
    """ Simple implementation of a circular array.
        You can append to it any number of times but only "size" items will be kept
    """
    def __init__(self, size):
        self.arr = np.zeros(size)
        self.ind = 0
        self.num_els = 0

    def append(self, value):
        if self.num_els < self.arr.shape[0]:
            self.num_els += 1
        self.arr[self.ind] = value
        self.ind = (self.ind + 1) % self.arr.shape[0]

    def mean(self):
        return np.mean(self.arr[:self.num_els])

    def median(self):
        return np.median(self.arr[:self.num_els])

class Timer:
    """ Simple helper class to compute the rate at which something is called.
        
        "smoothing" determines the size of the underlying circular array, which averages
        out variations in call rate over time.

        use timer.tick() to record an event
        use timer.fps() to report the average event rate.
    """
    def __init__(self, smoothing):
        self.arr = CircularArray(smoothing)
        self.last_time = time.time()

    def tick(self):
        t = time.time()
        self.arr.append(1.0 / (t - self.last_time))
        self.last_time = t

    def fps(self):
        return self.arr.mean()

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    q = tf_transformations.quaternion_from_euler(0, 0, angle)
    q_out = Quaternion()
    q_out.x = q[0]
    q_out.y = q[1]
    q_out.z = q[2]
    q_out.w = q[3]
    return q_out

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf_transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def pose_to_matrix(pose):
    """Return the homogeneous transform represented by a ROS Pose."""
    translation = tf_transformations.translation_matrix([
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ])
    rotation = tf_transformations.quaternion_matrix([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    return tf_transformations.concatenate_matrices(translation, rotation)


def transform_to_matrix(transform):
    """Return the homogeneous transform represented by a ROS Transform."""
    translation = tf_transformations.translation_matrix([
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ])
    rotation = tf_transformations.quaternion_matrix([
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ])
    return tf_transformations.concatenate_matrices(translation, rotation)


def map_to_odom_matrix(
        map_to_laser, odom_to_base, base_to_laser_transform):
    """Compute map -> odom while preserving the localized laser pose."""
    odom_to_laser = tf_transformations.concatenate_matrices(
        odom_to_base, transform_to_matrix(base_to_laser_transform))
    return tf_transformations.concatenate_matrices(
        map_to_laser, tf_transformations.inverse_matrix(odom_to_laser))


def rotation_matrix(theta):
    ''' Creates a rotation matrix for the given angle in radians '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

def particle_to_pose(particle):
    ''' Converts a particle in the form [x, y, theta] into a Pose object '''
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particles_to_poses(particles):
    ''' Converts a two dimensional array of particles into an array of Poses. 
        Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
    '''
    return list(map(particle_to_pose, particles))

# DEPRECATED: should make the header inside the node now
# def make_header(frame_id, stamp=None):
#     ''' Creates a Header object for stamped ROS objects '''
#     if stamp == None:
#         stamp = rospy.Time.now()
#     header = Header()
#     header.stamp = stamp
#     header.frame_id = frame_id
#     return header

def map_to_world_slow(x,y,t,map_info):
    ''' Converts given (x,y,t) coordinates from the coordinate space of the map (pixels) into world coordinates (meters).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the map_to_world function
    ''' 
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    map_c = np.array([[x],
                      [y]])
    world = (rot*map_c) * scale + trans

    return world[0, 0], world[1, 0], wrap_angle(t + angle)

def map_to_world(poses, map_info):
    ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from map coordinate space (pixels) to world coordinate space (meters).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as map_to_world_slow but vectorized and inlined
    '''

    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]

    # scale
    poses[:,:2] *= float(scale)

    # translate
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:, 2] = wrap_angle(poses[:, 2] + angle)

def world_to_map(poses, map_info):
    ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from world coordinate space (meters) to world coordinate space (pixels).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as world_to_map_slow but vectorized and inlined
        - You may have to transpose the returned x and y coordinates to directly index a pixel array
    '''
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y

    # scale
    poses[:,:2] *= (1.0/float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]
    poses[:, 2] = wrap_angle(poses[:, 2] + angle)

def world_to_map_slow(x,y,t, map_info):
    ''' Converts given (x,y,t) coordinates from the coordinate space of the world (meters) into map coordinates (pixels).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the world_to_map function
    ''' 
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(-angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    world = np.array([[x],
                      [y]])
    map_c = rot*((world - trans) / float(scale))
    return map_c[0, 0], map_c[1, 0], wrap_angle(t - angle)
