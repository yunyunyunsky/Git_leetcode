# 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那 两个 整数，并返回它们的数组下标。
#
#  你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
#
#  你可以按任意顺序返回答案。
#
#
#
#  示例 1：
#
#
# 输入：nums = [2,7,11,15], target = 9
# 输出：[0,1]
# 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
#
#
#  示例 2：
#
#
# 输入：nums = [3,2,4], target = 6
# 输出：[1,2]
#
#
#  示例 3：
#
#
# 输入：nums = [3,3], target = 6
# 输出：[0,1]
#
#
#
#
#  提示：
#
#
#  2 <= nums.length <= 10⁴
#  -10⁹ <= nums[i] <= 10⁹
#  -10⁹ <= target <= 10⁹
#  只会存在一个有效答案
#
#
#
#
#  进阶：你可以想出一个时间复杂度小于 O(n²) 的算法吗？
#
#  Related Topics 数组 哈希表 👍 20771 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 哈希求解,或者可以优化成查找之前的
        hashmap = {}
        for index, num in enumerate(nums):
            hashmap[num] = index
        for i, num in enumerate(nums):
            j = hashmap.get(target - num)
            if j is not None and i != j:
                return [i, j]

# leetcode submit region end(Prohibit modification and deletion)

# # leetcode submit region begin(Prohibit modification and deletion)
# class Solution(object):
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         # 遍历列表
#         for i in range(len(nums)):
#             # 计算需要找到的下一个目标数字
#             res = target - nums[i]
#             # 遍历剩下元素看是否存在该数字
#             if res in nums[i + 1:]:
#                 # 存在则返回
#                 return [i, nums[i + 1:].index(res) + i + 1]
#
# # leetcode submit region end(Prohibit modification and deletion)
