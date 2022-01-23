import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs = env.reset()

    '''
    Environment of cart pole are composed of 4 float numbers
        - x coordinate of stick's center of mass
        - speed
        - angle of poles to platform
        - angular speed
    '''
    print("Observation Space after reset :", obs)


    ''' 
    Action space here have 2 discrete value, 0 and 1.
    0 = move platform to the left, 1 = move platform to the right
    '''
    print("Action space : ", env.action_space)
    print("Total number of observation space :", env.observation_space)

    print("State after move platfor to the left", env.step(0))

