


import json
import os
import googleapiclient.discovery
import googleapiclient.errors

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def search(query):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    developerKey = "AIzaSyDZcSD1fC_OjUKI9kmS_L0K0EZ9qEGafKk"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=developerKey)

    videos = []
    if query:
        print("Query: ", query)
        request = youtube.search().list(
            part="snippet",
            q=query,
            maxResults=15,
            type='video',
            videoCaption='any'
        )
        response = request.execute()
        print("Got search response...")

        ids = ''
        for items in response['items']:
            ids += ', '
            ids += items['id']['videoId']
            videos.append(items['id']['videoId'])
        ids = ids[1:] #remove comma in starting

        # Downloads the statistics and content details about the searched videos
        video_request = youtube.videos().list(
            part="snippet, contentDetails, statistics",
            id=ids
        )

        video_details = video_request.execute()

        for i in range(len(video_details['items'])):
            response['items'][i]['rating'] = video_details['items'][i]

        # with open(f'video_ids/{query}.json', 'w+') as file:
        #     json.dump(response, file)
        #     file.close()

        # print("Saved search response as", f'{query}.json')
    print("list of searched videos", videos)
    return videos


if __name__ == '__main__':
    search(str(input("Search :")))

