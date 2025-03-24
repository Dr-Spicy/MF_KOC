import json
import os
import glob
import argparse
def merge_json_files(input_folder, output_folder, is_ex):
    """
    Merge JSON files from a folder (and subfolders) into a single JSON array.
    
    Args:
        input_folder (str): Path to the input folder containing JSON files.
        output_file (str): Path to foler containing the output JSON file.
    """
    if is_ex == "n":
        # Find all JSON files starting with 'content_name_' recursively
        pattern = os.path.join(input_folder, '**', 'creator_creator_*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        combined_data = []
        user_id_collection = []
        count_creator = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for i in data:
                        count_creator += 1
                        user_id = i["user_id"]
                        if user_id not in user_id_collection:
                            combined_data.append(i)
                            user_id_collection.append(user_id)
                print(f"Successfully processed: {json_file}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        file_name = "creator_creator_total.json"
        output_file = os.path.join(output_folder, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4,ensure_ascii=False)
        print(f"Successfully selected {len(user_id_collection)} unique creators from {count_creator} creators.")
        print(f"Successfully merged {len(json_files)} JSON files into {output_file}\n")

        pattern = os.path.join(input_folder, '**', 'creator_contents_*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        combined_data = []
        note_id_collection = []
        count_note = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for i in data:
                        count_note += 1
                        note_id = i.get("note_id", "")
                        if note_id not in note_id_collection:
                            liked_count = i.get("liked_count", "")
                            collected_count = i.get("collected_count", "")
                            comment_count = i.get("comment_count", "")
                            share_count = i.get("share_count", "")
                            try:
                                int(liked_count)
                                int(collected_count)
                                int(comment_count)
                                int(share_count)
                                combined_data.append(i)
                                note_id_collection.append(note_id)
                            except:
                                print(os.path.basename(json_file),i.get("nickname", "")+" of "+note_id+" :"+liked_count,collected_count,comment_count,share_count)
                                
                print(f"Successfully processed: {json_file}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        file_name = "creator_contents_total.json"
        output_file = os.path.join(output_folder, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4,ensure_ascii=False)
        print(f"Successfully selected {len(note_id_collection)} unique notes from {count_note} notes." )
        print(f"Successfully merged {len(json_files)} JSON files into {output_file}\n")
    elif is_ex == "y":
        pattern = os.path.join(input_folder, '**/*.json')
        json_files = glob.glob(pattern, recursive=True)
        combined_data = []
        note_id_collection = []
        count_note = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for i in data:
                        count_note += 1
                        note_id = i.get("note_id", "")
                        if note_id not in note_id_collection:
                            image = ""
                            taglist = ""
                            for j in i.get("image_list", ""):
                                image += j.get("info_list", "")[1].get("url","")
                                image += ","
                            for j in i.get("tag_list", ""):
                                taglist += j.get("name", "")
                                taglist += ","
                            dict = {
                                "note_id": note_id,
                                "type": i.get("type", ""),
                                "title": i.get("title", ""),
                                "desc": i.get("desc", ""),
                                "video_url": "",
                                "time": i.get("time", ""),
                                "last_update_time": i.get("last_update_time", ""),
                                "user_id": i.get("user", {}).get("user_id", ""),
                                "nickname": i.get("user", {}).get("nickname", ""),
                                "avatar": i.get("user", {}).get("avatar", ""),
                                "liked_count": i.get("interact_info", {}).get("liked_count", 0),
                                "collected_count": i.get("interact_info", {}).get("collected_count", 0),
                                "comment_count": i.get("interact_info", {}).get("comment_count", 0),
                                "share_count": i.get("interact_info", {}).get("share_count", 0),
                                "ip_location": i.get("ip_location", ""), 
                                "image_list": image,
                                "tag_list": taglist,
                                "last_modify_ts": i.get("last_update_time", ""),
                                "note_url": "",
                                "source_keyword": "",
                                "xsec_token": i.get("user", {}).get("xsec_token", ""),
                            }
                            combined_data.append(dict)
                            note_id_collection.append(note_id)
                print(f"Successfully processed: {json_file}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        file_name = "creator_contents_extension.json"
        output_file = os.path.join(output_folder, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4,ensure_ascii=False)
        print(f"Successfully selected {len(note_id_collection)} unique notes from {count_note} notes." )
        print(f"Successfully merged {len(json_files)} JSON files into {output_file}\n")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Merge JSON files')
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing JSON files')
    parser.add_argument('--output', '-o', required=True,
                        help='Output JSON file folder path')
    parser.add_argument('--extension', '-e', required=True,
                        help='used for clean JSON files from Google chrome extension')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a valid directory")
        exit(1)
    
    merge_json_files(args.input, args.output, args.extension)