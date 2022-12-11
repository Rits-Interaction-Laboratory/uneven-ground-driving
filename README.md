# 不整地走行シミュレータ

本プロジェクトは不整地走行におけるより良い進行方向を提案するアプリケーションです。

## 開発

### 開発環境

- Ubuntu 20.04
- ROS Noetic
- Gazebo

### インストール

依存関係のインストール手順は各自で調べてください。

```sh
$ git clone <this repo>
$ cd uneven-ground-driving

$ cd src
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ../ && catkin_make
$ source devel/setup.zsh # シェルに応じて変更必須
```

### Gazebo 上でロボットをシミュレートする

```sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
$ roslaunch simulator teleop_key.launch
```

### データセットを作成する

まずは画像の保存先を作成してください。

```sh
$ mkdir ~/.ros/images
```

次に、下記コマンドで不整地走行をシミュレートしましょう。

計測データは`~/.ros`に保存されます。

```sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world_headless.launch
$ roslaunch simulator make_dataset.launch
```
