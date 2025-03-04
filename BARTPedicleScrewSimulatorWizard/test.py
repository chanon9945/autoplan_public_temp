from RobotInit import Robot

def __main__():
    robot = Robot(1000,100)
    print(robot.link_2.get_transform(100))

__main__()