# Ad Display/Click Data on Taobao.com

Ad Display/Click Data on Taobao.com 是阿里巴巴提供的一个淘宝展示广告点击率预估数据集。

地址： https://tianchi.aliyun.com/dataset/dataDetail?dataId=56&lang=en-us

## 数据集介绍

| 数据名称    | 说明 | 属性
| ---------- | --- | ---
| raw_sample | 原始的样本骨架 | 用户ID，广告ID，时间，资源位，是否点击
| ad_feature | 广告的基本信息 | 广告ID，广告计划ID，类目ID，品牌ID
| user_profile| 用户的基本信息| 用户ID，年龄层，性别等
| raw_behavior_log|用户的行为日志 |用户ID，行为类型，时间，商品类目ID，品牌ID

### 原始样本骨架 raw_sample

我们从淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录），构成原始的样本骨架。
字段说明如下：
```
(1) user_id：脱敏过的用户ID；
(2) adgroup_id：脱敏过的广告单元ID；
(3) time_stamp：时间戳；
(4) pid：资源位；
(5) noclk：为1代表没有点击；为0代表点击；
(6) clk：为0代表没有点击；为1代表点击；
```
我们用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）。

### 广告基本信息表 ad_feature
本数据集涵盖了raw_sample中全部广告的基本信息。字段说明如下：
```
(1) adgroup_id：脱敏过的广告ID；
(2) cate_id：脱敏过的商品类目ID；
(3) campaign_id：脱敏过的广告计划ID；
(4) customer_id:脱敏过的广告主ID；
(5) brand：脱敏过的品牌ID；
(6) price: 宝贝的价格
```
其中一个广告ID对应一个商品（宝贝），一个宝贝属于一个类目，一个宝贝属于一个品牌。

### 用户基本信息表 user_profile
本数据集涵盖了raw_sample中全部用户的基本信息。字段说明如下：
```
(1) userid：脱敏过的用户ID；
(2) cms_segid：微群ID；
(3) cms_group_id：cms_group_id；
(4) final_gender_code：性别 1:男,2:女；
(5) age_level：年龄层次；
(6) pvalue_level：消费档次，1:低档，2:中档，3:高档；
(7) shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
(8) occupation：是否大学生 ，1:是,0:否
(9) new_user_class_level：城市层级
```

### 用户的行为日志 behavior_log
本数据集涵盖了raw_sample中全部用户22天内的购物行为(共七亿条记录)。字段说明如下：
```
(1) user：脱敏过的用户ID；
(2) time_stamp：时间戳；
(3) btag：行为类型,  包括以下四种：
    |类型    | 说明
    | ----- | ---
    |ipv | 浏览
    |cart | 加入购物车
    |fav | 喜欢
    |buy | 购买
(4) cate：脱敏过的商品类目；
(5) brand: 脱敏过的品牌词；
```

这里以user + time_stamp为key，会有很多重复的记录；这是因为我们的不同的类型的行为数据是不同部门记录的，在打包到一起的时候，实际上会有小的偏差（即两个一样的time_stamp实际上是差异比较小的两个时间）。

## 基线
AUC：0.622
