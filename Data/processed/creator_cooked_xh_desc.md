# Final Dataset Overview: `creator_cooked_xh.json`

This dataset provides a multidimensional view of creator performance by integrating detailed profile information with a variety of aggregated engagement and temporal metrics. Each row represents a unique creator and includes both static attributes and dynamic performance measures computed over different time windows. Data is combined from the rednote/xiaohongshu's database (1-16) and xinhong website.

## Columns Description

1. **note_id**  
   - *Description:* Unique identifier for each post.  
   - *Type:* String.

2. **content_type_video**  
   - *Description:* Indicator or label specifying whether the post is a video post. This field differentiates video content from other types. 1 means it is video post and 0 means it is normal post. 
   - *Type:* Integer.

3. **title**  
   - *Description:* Title of the post.  
   - *Type:* String.

4. **desc**  
   - *Description:* Description or body text of the post.  
   - *Type:* String.

5. **post_time**  
   - *Description:* Time representing when the post was created.  
   - *Type:* Datetime.

6. **last_update_time**  
   - *Description:* Time indicating the last time the post was updated.  
   - *Type:* Datetime

7. **user_id**  
   - *Description:* Identifier linking the post to its creator.  
   - *Type:* String.

8. **liked_count**  
   - *Description:* Number of likes the post has received.  
   - *Type:* Integer.

9. **collected_count**  
   - *Description:* Number of times the post has been saved or bookmarked.  
   - *Type:* Integer.

10. **comment_count**  
    - *Description:* Number of comments on the post.  
    - *Type:* Integer.

11. **share_count**  
    - *Description:* Number of shares the post has received.  
    - *Type:* Integer.

12. **tag_list**  
    - *Description:* JSON-formatted string containing additional tags or metadata associated with the post.  
    - *Type:* String.

13. **image_count**  
    - *Description:* Count of image URLs associated with the post.  
    - *Type:* Integer.

14. **interaction_count**  
    - *Description:* A computed metric representing the total interactions (e.g., sum of likes, collections, and shares) for the post.  
    - *Type:* Integer.

15. **hot_note**  
    - *Description:* Indicator or metric denoting whether the post is considered popular or "hot". 1 means it is hot and 0 means it is not.  
    - *Type:* Integer.

16. **extract_ts**  
    - *Description:* Time used for content extraction or indicating when the data was processed.  
    - *Type:* Datetime.

17. **图文报价(RMB)**  
   - *Description:* 新红网站所给出的creator评估价格，单位为人民币
   - *Type:* Integer.

18. **图文报价有对号**  
   - *Description:* 该名creator是否进行过商业合作，1代表是，0代表否
   - *Type:* Integer.

19. **平台等级**  
   - *Description:* 小红书对于creator评估的平台等级从最低1到最高10，等级9和等级10被归类为等级9，未知等级被推测为等级3
   - *Type:* Integer.

20. **活跃粉丝占比(仅>1K)**  
   - *Description:* 新红网站调查得到的creator粉丝中活跃粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

21. **粉丝女性比例**  
   - *Description:* 新红网站调查得到的creator粉丝中女性粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

22. **粉丝年龄<18**  
   - *Description:* 新红网站调查得到的creator粉丝中小于18岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

23. **粉丝年龄18-24**  
   - *Description:* 新红网站调查得到的creator粉丝中大于18岁且小于24岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

24. **粉丝年龄25-34**  
   - *Description:* 新红网站调查得到的creator粉丝中大于25岁且小于34岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

25. **粉丝年龄35-44**  
   - *Description:* 新红网站调查得到的creator粉丝中大于35岁且小于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.

26. **粉丝年龄>44**  
   - *Description:* 新红网站调查得到的creator粉丝中大于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null
   - *Type:* Float or string.
   
27. **兴趣标签**  
   - *Description:* 新红网站调查得到的creator发帖倾向标签的前五名，每行记录一个标签的排名、标签内容和占比
   - *Type:* string.

28. **地域分布**  
   - *Description:* 新红网站调查得到的creator粉丝地理分布的前五名，每行记录一个地域的排名、地域内容和占比
   - *Type:* string.