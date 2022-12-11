# 不整地走行シミュレータ

本プロジェクトは不整地走行におけるより良い進行方向を提案するアプリケーションです。

## 開発

### 開発環境

- Ubuntu 20.04
- ROS Noetic
- Gazebo
- Python 3.9

### インストール

まずはROS Noeticをインストールしてください。

未確認ですが、別ディストリビューションでも動くかもしれません。

```sh
$ sudo apt install -y ros-noetic-desktop-full
```

次に本リポジトリをcloneし、catkinプロジェクトをビルドしましょう。

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

## 実行方法

### Gazebo 上でロボットをシミュレートする

```sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
$ roslaunch simulator teleop_key.launch
```

### データセットを作成する

計測データは`~/.ros/uneven-ground-driving-result`に保存されます。

```sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world_headless.launch
$ GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH:$HOME/workspace/uneven-ground-driving" roslaunch turtlebot3_gazebo turtlebot3_dem_world.launch

# 別のターミナルでデータセット作成スクリプトを実行
$ roslaunch simulator make_dataset.launch
```