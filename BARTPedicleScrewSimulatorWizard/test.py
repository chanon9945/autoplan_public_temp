from RobotInit import Robot

def __main__():
    robot = Robot(1000,100)
    print(robot.link_1.transform[:,:,1])

__main__()