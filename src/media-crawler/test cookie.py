import requests
headers = {
    "Cookie": "abRequestId=97b847d1-b4ae-5803-a58c-890f8e95c982; webBuild=4.59.0; a1=19578fbf5a1ehgksn9hzeznf7xlm4ejs135ef1wkm50000344533; webId=e1dc600cf338c83b8d87ee142a55cfee; gid=yj2WYiS80YyWyj2WYiDi2qyE0ydxkTMIjxud2T6uIiWC1d2864dVMi888q442qq8ji0Kjidy; web_session=040069b649c138564f263527fa354ba9baee62; xsecappid=xhs-pc-web; websectiga=cf46039d1971c7b9a650d87269f31ac8fe3bf71d61ebf9d9a0a87efb414b816c; sec_poison_id=38bf69d9-e006-43ba-9345-7081b2bc4827; acw_tc=0a4a8a3717417457365086222e547e461c307599184f5e43145d5cc7dd40d7; loadts=1741745984818; unread={%22ub%22:%2267c61baf00000000280346ef%22%2C%22ue%22:%2267bb30b7000000002901046b%22%2C%22uc%22:17}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
}
response = requests.get("https://www.xiaohongshu.com/explore", headers=headers)
print(response.status_code)  # 200为正常，403表示Cookie失效