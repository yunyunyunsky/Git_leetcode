#
# DataFrame players:
# +-------------+--------+
# | Column Name | Type   |
# +-------------+--------+
# | player_id   | int    |
# | name        | object |
# | age         | int    |
# | position    | object |
# | ...         | ...    |
# +-------------+--------+
#
#
#  编写一个解决方案，计算并显示 players 的 行数和列数。
#
#  将结果返回为一个数组：
#
#  [number of rows, number of columns]
#
#  返回结果格式如下示例所示。
#
#
#
#  示例 1：
#
#
# 输入：
# +-----------+----------+-----+-------------+--------------------+
# | player_id | name     | age | position    | team               |
# +-----------+----------+-----+-------------+--------------------+
# | 846       | Mason    | 21  | Forward     | RealMadrid         |
# | 749       | Riley    | 30  | Winger      | Barcelona          |
# | 155       | Bob      | 28  | Striker     | ManchesterUnited   |
# | 583       | Isabella | 32  | Goalkeeper  | Liverpool          |
# | 388       | Zachary  | 24  | Midfielder  | BayernMunich       |
# | 883       | Ava      | 23  | Defender    | Chelsea            |
# | 355       | Violet   | 18  | Striker     | Juventus           |
# | 247       | Thomas   | 27  | Striker     | ParisSaint-Germain |
# | 761       | Jack     | 33  | Midfielder  | ManchesterCity     |
# | 642       | Charlie  | 36  | Center-back | Arsenal            |
# +-----------+----------+-----+-------------+--------------------+
# 输出：
# [10, 5]
# 解释：
# 这个 DataFrame 包含 10 行和 5 列。
#
#
#  👍 4 👎 0
import pandas as pd

def getDataframeSize(players: pd.DataFrame) ->list:
    return [players.shape[0], players.shape[1]]

data = [['1', 'Bob', '15', 'Bob', 'pandas'],
        ['2', 'jack', '12', 'jack', 'numpy'],
        ['3', 'jack', '12', 'jack', 'numpy']]
players = pd.DataFrame(data,
                        columns=['id', 'name', 'age', 'name', 'style'],
                       index=['小红', '小李', '小刚'])
print(players)
print(getDataframeSize(players))

# There is no code of Python type for this problem