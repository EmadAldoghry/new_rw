# Published by the image_segmentation_node, containing detected pixels.
std_msgs/Header header

# Arrays of u (horizontal) and v (vertical) pixel coordinates.
# Should be of the same length.
uint32[] u
uint32[] v