import glob
import os
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt
import math
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.set_soft_device_placement = False
tf.config.experimental.set_memory_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=4096)])

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus), 'Logical gpus')

with tf.device('/device:GPU:0'):
    w = tf.constant([[2, -3.4]])
    b = tf.constant([4.2])
    x = tf.random.normal([1000, 2], mean=0, stddev=10)
    e = tf.random.normal([1000, 2], mean=0, stddev=0.1)
    W = tf.Variable(tf.constant([5, 1]))
    B = tf.Variable(tf.constant([1]))

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False
IM_WIDTH = 80
IM_HEIGHT = 60
EPISODES = 100

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    #collision_hist = []
    actor_list = []
    actor_list1 = []


    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.load_world('Town06')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.transform = carla.Transform(carla.Location(x=20, y=150, z=1))
        self.transform1 = carla.Transform(carla.Location(x=100, y=150, z=1))





    def reset(self, max_epoch_time=None):
        self.collision_hist = []

        try:
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            self.actor_list.append(self.vehicle)
            self.vehicle1 = self.world.spawn_actor(self.model_3, self.transform1)
            self.actor_list1.append(self.vehicle1)

            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
            self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
            self.rgb_cam.set_attribute("fov", f"110")

            transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
            self.actor_list.append(self.sensor)
            self.sensor.listen(lambda data: self.process_img(data))

            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))

            while self.front_camera is None:
                time.sleep(0.01)

            self.episode_start = time.time()
            self.start_time = dt.datetime.now()
            self.end_time = None
            if max_epoch_time:

                self.expected_end_time = self.start_time + \
                                     dt.timedelta(seconds=max_epoch_time)
            else:
                self.expected_end_time = None

            return self.front_camera
        except RuntimeError:
            print('respawn actor')
            time.sleep(10)
            env.reset()



    def collision_data(self, event):
        self.collision_hist.append(event)


    def process_img(self,image):

        i1 = np.array(image.raw_data)
        i2 = i1.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.front_camera = i3

    def get_velocity(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        return kmh

    def get_location(self):
        l = self.vehicle.get_location()
        L = int(l.x - 20)
        return L



    def control(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0))
            return 0.5, 0
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, brake= 0))
            return 0.75, 0
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake= 0))
            return 1.0, 0
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake= 0.5))
            return 0, 0.5
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake= 1.0))
            return 0, 1.0

    def get_reward(self,action):
        l = self.vehicle.get_location()
        index = np.argmin(abs(l.x - file['X']))
        if action == 0:
            reward = -0.5 * abs(0.5-file['THROTTLE'][index])-0.5 * abs(0-file['BRAKE'][index])

        elif action == 1:
            reward = -0.5 * abs(0.75 - file['THROTTLE'][index]) - 0.5 * abs(0 - file['BRAKE'][index])

        elif action == 2:
            reward = -0.5 * abs(1.0 - file['THROTTLE'][index]) - 0.5 * abs(0 - file['BRAKE'][index])

        elif action == 3:
            reward = -0.5 * abs(0 - file['THROTTLE'][index]) - 0.5 * abs(0.5 - file['BRAKE'][index])

        elif action == 4:
            reward = -0.5 * abs(0 - file['THROTTLE'][index]) - 0.5 * abs(1 - file['BRAKE'][index])
        else:
            reward = 0



        if len(self.collision_hist) != 0:
            done = True
        else:
            done = False

        if self.episode_start + max_epoch_time < time.time():
            done = True
        return  reward, done, None


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in \
                self.memory.columns)


class DQNAgent():
    def __init__(self, gamma=0.99, batch_size=32,
                 replayer_capacity=2000, random_initial_steps=50,
                 weight_path=None, train_conv=True,
                 epsilon=1., min_epsilon=0.1, epsilon_decrease_rate=0.003):
        self.action_n = 5
        self.gamma = gamma

        # 经验回放
        self.replayer = DQNReplayer(capacity=replayer_capacity)
        self.batch_size = batch_size
        self.random_initial_steps = random_initial_steps

        # 探索参数
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decrease_rate = epsilon_decrease_rate

        # 搭建网络
        self.evaluate_net = self.build_network(weight_path=weight_path,
                                               train_conv=train_conv)
        self.target_net = self.build_network()
        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, activation='relu', weight_path=None,
                      train_conv=True, verbose=True):
        inputs = keras.Input(shape=(80, 60, 3))
        x = inputs

        # 卷积层
        for filte in [16, 32, 32]:
            z = keras.layers.Conv2D(filte, 3, padding='same',
                                    activation=activation,
                                    trainable=train_conv)(x)
            x = keras.layers.MaxPooling2D(pool_size=2)(z)

        y = keras.layers.Flatten()(x)

        # 全连接层
        x = keras.layers.Dropout(0.2)(y)
        z = keras.layers.Dense(128, activation=tf.nn.relu,
                               kernel_initializer=RandomNormal(stddev=0.01))(x)
        y = keras.layers.Dropout(0.2)(z)
        outputs = keras.layers.Dense(self.action_n,
                                     kernel_initializer=RandomNormal(stddev=0.01))(y)

        net = keras.Model(inputs=inputs, outputs=outputs)
        net.compile(optimizer='adam', loss='mse')

        if verbose:
            net.summary()

        if weight_path:
            net.load_weights(weight_path)
            if verbose:
                print('载入网络权重 {}'.format(weight_path))

        return net

    def decide(self, observation, random=False):
        if random or np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        observations = observation[np.newaxis]
        qs = self.evaluate_net.predict(observations)
        return np.argmax(qs)


    def learn(self, observation, action, reward, next_observation, done):
        agent.replayer.store(observation, action, reward, next_observation,
                             done)  # 存储经验

        if self.replayer.count < self.random_initial_steps:
            return  # 还没到存足够多的经验，先不训练神经网络

        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)  # 经验回放


        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())
            self.target_net.save('target_model_new_manual1.h5')

        # 减小 epsilon 的值
        self.epsilon -= self.epsilon_decrease_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)


def play_once(env, agent, random=False, train=False, max_epoch_time=None, wait_delta_sec=0.01, verbose=True):
    env.reset(max_epoch_time)
    total_reward = 0

    # 正式开始学习
    for step in itertools.count():

        image = env.front_camera
        #cv2.imshow('', image)
        car_state = env.get_velocity()
        action = agent.decide(image, random=random)

        # 根据动作影响环境
        throttle, brake = env.control(action)
        his = len(env.collision_hist)
        distance = env.get_location()
        if verbose:
            print('动作 = {}, 速度 = {}, 油门 = {}, 刹车 = {}, 距离 = {}, 历史 = {}' \
                  .format(action, car_state, throttle, brake, distance, his))

        # 等待一段时间
        time.sleep(wait_delta_sec)

        # 获得更新后的观测、奖励和回合结束指示
        next_image = env.front_camera
        reward, done, info = env.get_reward(action)
        total_reward += reward

        # 如果回合刚开始就结束了，就不是靠谱的回合
        if step == 0 and done:
            if verbose:
                print('不成功的回合，放弃保存')
                for actor in env.actor_list[-3:]:
                    actor.destroy()
                    print('actor cleaned up')
                for actor1 in env.actor_list1[-1:]:
                    actor1.destroy()
                    print('actor1 cleaned up')
            break

        if train:  # 根据经验学习
            agent.learn(image, action, reward, next_image, done)

        # 回合结束
        if done:
            if verbose:
                print('回合 从 {} 到 {} 结束. 总奖励：{}'.format(
                    env.start_time, env.end_time, total_reward))
                for actor in env.actor_list[-3:]:
                    actor.destroy()
                    print('actor cleaned up')
                for actor1 in env.actor_list1[-1:]:
                    actor1.destroy()
                    print('actor1 cleaned up')
            return total_reward


weight_path = None # 载入权重数据的位置
train_conv = False # 是否训练卷积层
max_epoch_time = 15. # 最长回合时间
random_initial_steps = 1000 # 随机运行的初始步数
train = True # 是否训练


env = CarEnv()
agent = DQNAgent(weight_path=weight_path, train_conv=train_conv,
        random_initial_steps=random_initial_steps)

file = pd.read_csv('manual_data.csv')
df = pd.DataFrame(file)

if train:
    print('开始训练')
    reward_sum = []
    while True: # 无限循环，永不停止。需要手动中断

        random = agent.replayer.count < random_initial_steps
        r = play_once(env, agent, random=random, train=True)
        reward_sum.append(r)
        print(reward_sum)
    agent.target_net.save('target_model_manual.h5')
    plt.plot(reward_sum[1:])
    plt.xlabel('Number of iterations')
    plt.ylabel('reward_sum')

    plt.show()


else:
    print('开始测试')
    agent.epsilon = 0. # 取消探索
    play_once(env, agent, max_epoch_time=max_epoch_time)