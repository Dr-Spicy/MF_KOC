# Final Dataset Overview: `contents_cooked.json`

This dataset provides detailed information about individual posts (or "notes") created by content creators. Each row represents a single post and includes both raw metadata and computed engagement metrics that support content-level analysis.

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

