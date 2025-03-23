# 流程概览:
1. 在xhs上自行找到符合下面具体条件的KOC.
2. 到共享KOC list表上确认此KOC尚未被其他人认领
3. 复制粘贴一个未被认领KOC的profile链接到KOC list column C, 用column D的函数提取出KOC的user_id
4. 把user_id更新到base_config.py的 XHS_CREATOR_ID_LIST里面, 然后根据下面的设置开始抓取, 最开始时候建议每次抓一个user且headless=False, 流程稳定熟悉之后headless=True,每次抓2-3个user, VPN换IP, (if needed)换cookie
5. 根据creator_creator_日期.json的信息把KOC list表E-L列填上
6. 注册新红数据, 利用每天免费额度填上KOC list表K列以后的部分
7. **(重要)检查帖子的时候如果发现了10+这样的关键词, 不必重新抓取, 而是参考读出xhs笔记精确分享数.md来手动解决https://github.com/Dr-Spicy/KOC_MediaCrawler_Xiaohan/blob/main/%E8%AF%BB%E5%87%BAxhs%E7%AC%94%E8%AE%B0%E7%B2%BE%E7%A1%AE%E5%88%86%E4%BA%AB%E6%95%B0.md.**

**在xhs自选KOC rule:**
DFW本地周边
粉丝在9000和200之间, 最好是8000和500之间
有代表性, 最好在传统探店类外能有非垂直探店博主类, 混合探店博主类 非中文母语, 比如有一定粉丝基础的tk难民
可以从已经找到的KOC的热帖的评论区去顺藤摸瓜找其他的candidates
利用DFW本地学校等landmark

**如何使用KOC list表格:**
A列自己名字,
C列URL只保留?之前的部分(不然D列你用不了)
D列使用我定义的函数直接提取user_id
E-L从creator_creator_日期.json 或者 用户主页复制
抓取完成后灰色高亮
K列以后从新红数据(请自行去注册,利用每天的免费额度)复制
L列是新红图文合作报价, 如果是有对号(合作过而非预估), 请在 
最后在新红数据截图保存网红的笔记分析, 因为它会提供最近180天的帖子的阅读量(扒不下来)

**最新的xhs爬虫设置相关文件**
https://github.com/Dr-Spicy/KOC_MediaCrawler_Xiaohan

**步骤教程:**
(注意:按照下面流程并稍作修改)
git clone https://github.com/NanmiCoder/MediaCrawler.git
变成
git clone https://github.com/Dr-Spicy/KOC_MediaCrawler_Xiaohan.git


**一定要安装并用python3.9.6代替默认python interpretor, 细节看视频** 

**如何找到自己的cookie:**
登录xhs web后, 刷新出最新的homefeed里的request headers中寻找一个很长的cookie. 
一定要把自己cookie放入base_config.py 的cookie

**检查自己的cookie是否仍然有效:** 
在虚拟环境中输入 python 'test cookie.py', 如果输出200即有效

**如何开始爬:**
在虚拟环境中输入 python main.py --platform xhs --lt cookie --type creator

**数据在哪:**
/data/xhs/json文件夹中
结果会以天为单位保存在creator_creator.json and creator_contents.json. 请确保抓取后检查一下对KOC的帖子抓取是否完整,(检查一下新抓帖子的数量, 抽查帖子抓取的质量)


**抓取时的经验教训:**
1. 建议对于具体如何设置MediaCrawler运行环境和提取cookie有疑问的家人们, 可以参考我发的视频录屏
2. CRAWLER_MAX_SLEEP_SEC
设的大一些(geq20), 有流量追踪机制, 抓太快会被封IP几个小时, VPN设置到其他大洲(尤其是香港或者欧洲)或有帮助
3. 建议一次抓少于3个KOC, 然后换IP with VPN
4. 经常检查自己的cookie是否失效, 保质期可能不到几天, 甚至小于1天(after IP change)
5. 出现error 102说明被强烈风控或者cookie可能失效, 需要换小号或者换IP
6. 出现error 309说明系统在soft风控你, 需要在config中prolong CRAWLER_MAX_SLEEP_SEC, 并检查当前KOC的笔记是否抓取完整 by note_id. 但一般来说Error 309结束后XHS服务器再次回复通话流量时候会重复抓取之前没有抓到的帖子. 
7. 结果会以天为单位保存在creator_creator.json and creator_contents.json. 请确保抓取后检查一下对KOC的帖子抓取是否完整,(检查一下新抓帖子的数量, 抽查帖子抓取的质量)
8. **(重要)检查帖子的时候如果发现了10+这样的关键词, 请务必参考读出xhs笔记精确分享数.md来解决https://github.com/Dr-Spicy/KOC_MediaCrawler_Xiaohan/blob/main/%E8%AF%BB%E5%87%BAxhs%E7%AC%94%E8%AE%B0%E7%B2%BE%E7%A1%AE%E5%88%86%E4%BA%AB%E6%95%B0.md.**




# 🔥 自媒体平台爬虫🕷️MediaCrawler🔥 
<a href="https://trendshift.io/repositories/8291" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8291" alt="NanmiCoder%2FMediaCrawler | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![GitHub Stars](https://img.shields.io/github/stars/NanmiCoder/MediaCrawler?style=social)](https://github.com/NanmiCoder/MediaCrawler/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/NanmiCoder/MediaCrawler?style=social)](https://github.com/NanmiCoder/MediaCrawler/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/pulls)
[![License](https://img.shields.io/github/license/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/blob/main/LICENSE)

> **免责声明：**
> 
> 大家请以学习为目的使用本仓库⚠️⚠️⚠️⚠️，[爬虫违法违规的案件](https://github.com/HiddenStrawberry/Crawler_Illegal_Cases_In_China)  <br>
>
>本仓库的所有内容仅供学习和参考之用，禁止用于商业用途。任何人或组织不得将本仓库的内容用于非法用途或侵犯他人合法权益。本仓库所涉及的爬虫技术仅用于学习和研究，不得用于对其他平台进行大规模爬虫或其他非法行为。对于因使用本仓库内容而引起的任何法律责任，本仓库不承担任何责任。使用本仓库的内容即表示您同意本免责声明的所有条款和条件。
>
> 点击查看更为详细的免责声明。[点击跳转](#disclaimer)

# 仓库描述

**小红书爬虫**，**抖音爬虫**， **快手爬虫**， **B站爬虫**， **微博爬虫**，**百度贴吧爬虫**，**知乎爬虫**...。  
目前能抓取小红书、抖音、快手、B站、微博、贴吧、知乎等平台的公开信息。

原理：利用[playwright](https://playwright.dev/)搭桥，保留登录成功后的上下文浏览器环境，通过执行JS表达式获取一些加密参数
通过使用此方式，免去了复现核心加密JS代码，逆向难度大大降低

# 功能列表
| 平台   | 关键词搜索 | 指定帖子ID爬取 | 二级评论 | 指定创作者主页 | 登录态缓存 | IP代理池 | 生成评论词云图 |
| ------ | ---------- | -------------- | -------- | -------------- | ---------- | -------- | -------------- |
| 小红书 | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 抖音   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 快手   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| B 站   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 微博   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 贴吧   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 知乎   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |

### MediaCrawlerPro重磅发布啦！！！
> 主打学习成熟项目的架构设计，不仅仅是爬虫，Pro中的其他代码设计思路也是值得学习，欢迎大家关注！！！

[MediaCrawlerPro](https://github.com/MediaCrawlerPro) 版本已经重构出来了，相较于开源版本的优势：
- 多账号+IP代理支持（重点！）
- 去除Playwright依赖，使用更加简单
- 支持linux部署（Docker docker-compose）
- 代码重构优化，更加易读易维护（解耦JS签名逻辑）
- 代码质量更高，对于构建更大型的爬虫项目更加友好
- 完美的架构设计，更加易扩展，源码学习的价值更大
- Pro中新增全新的自媒体视频下载器桌面端软件（全栈项目适合学习）


# 安装部署方法
> 开源不易，希望大家可以Star一下MediaCrawler仓库！！！！十分感谢！！！ <br>

## 创建并激活 python 虚拟环境
> 如果是爬取抖音和知乎，需要提前安装nodejs环境，版本大于等于：`16`即可 <br>
   ```shell   
   # 进入项目根目录
   cd MediaCrawler
   
   # 创建虚拟环境
   # 我的python版本是：3.9.6，requirements.txt中的库是基于这个版本的，如果是其他python版本，可能requirements.txt中的库不兼容，自行解决一下。
   python -m venv venv
   
   # macos & linux 激活虚拟环境
   source venv/bin/activate

   # windows 激活虚拟环境
   venv\Scripts\activate

   ```

## 安装依赖库

   ```shell
   pip install -r requirements.txt
   ```

## 安装 playwright浏览器驱动

   ```shell
   playwright install
   ```

## 运行爬虫程序

   ```shell
   ### 项目默认是没有开启评论爬取模式，如需评论请在config/base_config.py中的 ENABLE_GET_COMMENTS 变量修改
   ### 一些其他支持项，也可以在config/base_config.py查看功能，写的有中文注释
   
   # 从配置文件中读取关键词搜索相关的帖子并爬取帖子信息与评论
   python main.py --platform xhs --lt qrcode --type search
   
   # 从配置文件中读取指定的帖子ID列表获取指定帖子的信息与评论信息
   python main.py --platform xhs --lt qrcode --type detail
  
   # 打开对应APP扫二维码登录
     
   # 其他平台爬虫使用示例，执行下面的命令查看
   python main.py --help    
   ```

## 数据保存
- 支持关系型数据库Mysql中保存（需要提前创建数据库）
    - 执行 `python db.py` 初始化数据库数据库表结构（只在首次执行）
- 支持保存到csv中（data/目录下）
- 支持保存到json中（data/目录下）



# 其他常见问题可以查看在线文档
> 
> 在线文档包含使用方法、常见问题、加入项目交流群等。
> [MediaCrawler在线文档](https://nanmicoder.github.io/MediaCrawler/)
> 

# 作者提供的知识服务
> 如果想快速入门和学习该项目的使用、源码架构设计等、学习编程技术、亦或者想了解MediaCrawlerPro的源代码设计可以看下我的知识付费栏目。

[作者的知识付费栏目介绍](https://nanmicoder.github.io/MediaCrawler/%E7%9F%A5%E8%AF%86%E4%BB%98%E8%B4%B9%E4%BB%8B%E7%BB%8D.html)

# 项目微信交流群

[加入微信交流群](https://nanmicoder.github.io/MediaCrawler/%E5%BE%AE%E4%BF%A1%E4%BA%A4%E6%B5%81%E7%BE%A4.html)
  
# 感谢下列Sponsors对本仓库赞助支持
- <a href="https://www.ipwo.net/?ref=mediacrawler">【IPWO住宅代理】免费流量测试，9000万+海外纯净真实住宅IP，全球覆盖，高品质代理服务提供商</a>
- <a href="https://sider.ai/ad-land-redirect?source=github&p1=mi&p2=kk">【Sider】全网最火的ChatGPT插件，我也免费薅羊毛用了快一年了，体验拉满。</a>

成为赞助者，可以将您产品展示在这里，每天获得大量曝光，联系作者微信：yzglan 或 email：relakkes@gmail.com


# 爬虫入门课程
我新开的爬虫教程Github仓库 [CrawlerTutorial](https://github.com/NanmiCoder/CrawlerTutorial) ，感兴趣的朋友可以关注一下，持续更新，主打一个免费.

# star 趋势图
- 如果该项目对你有帮助，帮忙 star一下 ❤️❤️❤️，让更多的人看到MediaCrawler这个项目

[![Star History Chart](https://api.star-history.com/svg?repos=NanmiCoder/MediaCrawler&type=Date)](https://star-history.com/#NanmiCoder/MediaCrawler&Date)


# 参考

- xhs客户端 [ReaJason的xhs仓库](https://github.com/ReaJason/xhs)
- 短信转发 [参考仓库](https://github.com/pppscn/SmsForwarder)
- 内网穿透工具 [ngrok](https://ngrok.com/docs/)


# 免责声明
<div id="disclaimer"> 

## 1. 项目目的与性质
本项目（以下简称“本项目”）是作为一个技术研究与学习工具而创建的，旨在探索和学习网络数据采集技术。本项目专注于自媒体平台的数据爬取技术研究，旨在提供给学习者和研究者作为技术交流之用。

## 2. 法律合规性声明
本项目开发者（以下简称“开发者”）郑重提醒用户在下载、安装和使用本项目时，严格遵守中华人民共和国相关法律法规，包括但不限于《中华人民共和国网络安全法》、《中华人民共和国反间谍法》等所有适用的国家法律和政策。用户应自行承担一切因使用本项目而可能引起的法律责任。

## 3. 使用目的限制
本项目严禁用于任何非法目的或非学习、非研究的商业行为。本项目不得用于任何形式的非法侵入他人计算机系统，不得用于任何侵犯他人知识产权或其他合法权益的行为。用户应保证其使用本项目的目的纯属个人学习和技术研究，不得用于任何形式的非法活动。

## 4. 免责声明
开发者已尽最大努力确保本项目的正当性及安全性，但不对用户使用本项目可能引起的任何形式的直接或间接损失承担责任。包括但不限于由于使用本项目而导致的任何数据丢失、设备损坏、法律诉讼等。

## 5. 知识产权声明
本项目的知识产权归开发者所有。本项目受到著作权法和国际著作权条约以及其他知识产权法律和条约的保护。用户在遵守本声明及相关法律法规的前提下，可以下载和使用本项目。

## 6. 最终解释权
关于本项目的最终解释权归开发者所有。开发者保留随时更改或更新本免责声明的权利，恕不另行通知。
</div>


## 感谢JetBrains提供的免费开源许可证支持
<a href="https://www.jetbrains.com/?from=MediaCrawler">
    <img src="https://www.jetbrains.com/company/brand/img/jetbrains_logo.png" width="100" alt="JetBrains" />
</a>

