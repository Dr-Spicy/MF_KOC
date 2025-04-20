# Final Dataset Overview: `creator_cooked_xh.json`

This dataset provides a multidimensional view of creator performance by integrating detailed profile information with a variety of aggregated engagement and temporal metrics. Each row represents a unique creator and includes both static attributes and dynamic performance measures computed over different time windows. Data is combined from the rednote/xiaohongshu's database (col:1-45) and xinhong website (col:46-57).

## Columns Description

1. **user_id**  
   - *Description:* Unique identifier for each creator.  
   - *Type:* String.

2. **nickname**  
   - *Description:* The display name of the creator.  
   - *Type:* String.

3. **avatar**  
   - *Description:* URL to the creator's profile image.  
   - *Type:* String (URL).

4. **desc**  
   - *Description:* Enriched description of the creator. This text combines the original profile description with appended tag-based insights (e.g., zodiac, age, profession, college).  
   - *Type:* String.

5. **ip_location**  
   - *Description:* Location inferred from the creator’s IP address.  
   - *Type:* String.

6. **follows**  
   - *Description:* Number of accounts the creator is following.  
   - *Type:* Integer.

7. **fans**  
   - *Description:* Number of followers the creator has.  
   - *Type:* Integer.

8. **interaction**  
   - *Description:* Overall engagement metric on the creator’s profile.  
   - *Type:* Integer.

9. **last_modify_ts**  
   - *Description:* Time when the creator’s profile or content was last updated (displayed in datetime format).  
   - *Type:* Datetime.

10. **pic_per_normal_note**  
    - *Description:* Average number of pictures per normal (non-video) post.  
    - *Type:* Float.

11. **video_ratio**  
    - *Description:* Ratio of video posts to total posts.  
    - *Type:* Float.

12. **hot_note_count**  
    - *Description:* Count of "hot" (popular) posts by the creator.  
    - *Type:* Integer.

13. **total_share_counts_hot_ratio**  
    - *Description:* Ratio of total share counts relative to hot posts, reflecting engagement on popular content.  
    - *Type:* Float.

14. **last_note2now**  
    - *Description:* number of days elapsed from the creator's most recent post to the current time.  
    - *Type:* Integer.

15. **last_hot_note2now**  
    - *Description:* number of days elapsed from the creator's most recent hot post to the current time.  
    - *Type:* Integer.

16. **weighted_total_share_counts**  
    - *Description:* Weighted sum of share counts across posts, emphasizing certain posts over others.  
    - *Type:* Float.

18. **liked_count**  
    - *Description:* Total number of likes received across all posts.  
    - *Type:* Integer.

19. **collected_count**  
    - *Description:* Total number of times posts were collected (saved or bookmarked).  
    - *Type:* Integer.

20. **comment_count**  
    - *Description:* Total number of comments received across all posts.  
    - *Type:* Integer.

21. **share_count**  
    - *Description:* Total number of shares accumulated from all posts.  
    - *Type:* Integer.

22. **note_count**  
    - *Description:* Total count of posts (notes) made by the creator.  
    - *Type:* Integer.

23. **location**  
    - *Description:* three Standardized location categories derived from tag information (TX, US, abroad).  
    - *Type:* String.

24. **ff_ratio**  
    - *Description:* Ratio of fans to follows, indicating the creator's influence.  
    - *Type:* Float.

25. **age_koc**  
    - *Description:* Account age in days, calculated as the difference between the newest and oldest posts.  
    - *Type:* Integer.

26. **is_female**  
    - *Description:* Binary indicator for gender; 1 indicates female, 0 indicates male.  
    - *Type:* Integer.

27. **min**  
    - *Description:* The minimum (earliest) post timestamp for the creator.  
    - *Type:* Datetime.

28. **max**  
    - *Description:* The maximum (latest) post timestamp for the creator.  
    - *Type:* Datetime.

29. **post_span**  
    - *Description:* Time span between the earliest and latest posts.  
    - *Type:* Timedelta or numeric representation (e.g., number of days).

30. **first_post_time**  
    - *Description:* Timestamp of the creator's first post.  
    - *Type:* Datetime.

31. **account_length**  
    - *Description:* Duration in days from the creator's first post to the last modification timestamp.  
    - *Type:* Integer.

32. **history_avg**  
    - *Description:* Average time interval between consecutive posts.  
    - *Type:* Float.

33. **history_std**  
    - *Description:* Standard deviation of the time intervals between posts.  
    - *Type:* Float.

34. **post_avg**  
    - *Description:* Average posting frequency (e.g., average number of posts per day).  
    - *Type:* Float.

35. **post_std**  
    - *Description:* Standard deviation of the posting frequency.  
    - *Type:* Float.

36. **liked_90**  
    - *Description:* Sum of likes received on posts made within the last 90 days (relative to the creator's last modification timestamp).  
    - *Type:* Float.

37. **collected_90**  
    - *Description:* Sum of collected counts from posts in the 90-day window.  
    - *Type:* Float.

38. **comment_90**  
    - *Description:* Sum of comments from posts in the last 90 days.  
    - *Type:* Float.

39. **share_90**  
    - *Description:* Sum of shares from posts within the 90-day period.  
    - *Type:* Float.

40. **note_count_90**  
    - *Description:* Count of posts made within the last 90 days.  
    - *Type:* Integer.

41. **liked_180**  
    - *Description:* Sum of likes received on posts over the past 180 days.  
    - *Type:* Float.

42. **collected_180**  
    - *Description:* Sum of collected counts from posts in the 180-day window.  
    - *Type:* Float.

43. **comment_180**  
    - *Description:* Sum of comments received on posts over the past 180 days.  
    - *Type:* Float.

44. **share_180**  
    - *Description:* Sum of shares from posts within the 180-day period.  
    - *Type:* Float.

45. **note_count_180**  
     - *Description:* Count of posts made within the last 180 days.  
     - *Type:* Integer.

46. **图文报价(RMB)**  
     - *Description:* 新红网站所给出的creator评估价格，单位为人民币
     - *Type:* Integer.

47. **图文报价有对号**  
     - *Description:* 该名creator是否进行过商业合作，1代表是，0代表否
     - *Type:* Integer.

48. **平台等级**  
     - *Description:* 小红书对于creator评估的平台等级从最低1到最高10，等级9和等级10被归类为等级9，未知等级被推测为等级3
     - *Type:* Integer.

49. **活跃粉丝占比(仅>1K)**  
     - *Description:* 新红网站调查得到的creator粉丝中活跃粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

50. **粉丝女性比例**  
     - *Description:* 新红网站调查得到的creator粉丝中女性粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

51. **粉丝年龄<18**  
     - *Description:* 新红网站调查得到的creator粉丝中小于18岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

52. **粉丝年龄18-24**  
     - *Description:* 新红网站调查得到的creator粉丝中大于18岁且小于24岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

53. **粉丝年龄25-34**  
     - *Description:* 新红网站调查得到的creator粉丝中大于25岁且小于34岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

54. **粉丝年龄35-44**  
     - *Description:* 新红网站调查得到的creator粉丝中大于35岁且小于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.

55. **粉丝年龄>44**  
     - *Description:* 新红网站调查得到的creator粉丝中大于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
     - *Type:* Float or string.
   
56. **兴趣标签**  
     - *Description:* 新红网站调查得到的creator发帖倾向标签的前五名，每行记录一个标签的排名、标签内容和占比
     - *Type:* string.

57. **地域分布**  
     - *Description:* 新红网站调查得到的creator粉丝地理分布的前五名，每行记录一个地域的排名、地域内容和占比
     - *Type:* string.


58. **score3_content_quality**  
     - *Description:* 旨在从多个维度综合衡量创作者内容的专业性、独特性、感染力和相关性。该指标通过整合四个关键维度的评分，为品牌识别优质内容创作者提供定量依据，确保投放资源能获得最佳营销效益。
     - *Type:* Float

    - A. **score3a_originality**
        - *Description:* 基于同自身内容(内部)和其他创作者内容(外部)比较后, 旨在定量评估创作者产出差异化内容的能力
        - *Type:* Float

    - B. **score3b_vertical**
        - *Description:* 通过与既定的目标领域集合对比, 定量分析创作者的内容专注度，有助于发现在目标领域持续有专业深度贡献的创作者
        - *Type:* Float
    
    - C. **score3c_sentiment**
        - *Description:* 通过NLP技术量化笔记中的情感特征，旨在识别那些能够传递丰富、真实情感体验，从而与受众产生深度共鸣的内容创作者
        - *Type:* Float

    - D. **score3d_keyword**
        - *Description:* 通过模糊文本匹配技术，识别那些持续产出与目标品牌/产品高度相关内容的创作者，确保推荐的KOC真正聚焦于特定主题
        - *Type:* Float