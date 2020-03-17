import requests
from bs4 import BeautifulSoup
import json

data_export = open("data_crawl.txt", "w")
data_source = open("sources.txt", "r")
data_lines = data_source.readlines()
for url_line in data_lines:
    i = 1
    status = True
    pageWithNumber = "page-"
    while i > 0 and status:
        if i == 1:
            pageWithNumber = ""
        else:
            pageWithNumber = "page-" + str(i)
        url = url_line + str(pageWithNumber)
        print('========================= Crawling ' + pageWithNumber + ' ==================================')
        # print(url)
        page = requests.get(url)

        if page.status_code != 200:
            # print("false")
            status = False
        else:
            soup = BeautifulSoup(page.content, 'html.parser')

            # start crawl quote
            quote_list = soup.find_all('div', class_='quote')
            for item in list(set(quote_list)):
                data_export.writelines(item.get_text() + '\n\n')
                # print("")

            soup_content = BeautifulSoup(page.content, 'html.parser')

            script_list = soup_content.find_all('script')
            for item in list(set(script_list)):
                content = item.get_text()
                if '__NEXT_DATA__' in content:
                    json_content = content.split(' = ')[1].split(';__')[0]
                    try:
                        data = json.loads(json_content)
                    except json.decoder.JSONDecodeError:
                        print(json_content)
                        break

                    pageProps = data['props']['pageProps']
                    api_data_comment = pageProps['apiData']

                    for key in api_data_comment:
                        posts_dict = api_data_comment[key]
                        if 'posts' in posts_dict:
                            for key_post in posts_dict:
                                if type(posts_dict[key_post]) == list:
                                    thread_post = posts_dict[key_post]
                                    for comment in thread_post:
                                        comment_content = comment['post_body']
                                        if '[CENTER]' not in comment_content:
                                            if '[/QUOTE]' in comment_content:
                                                data_export.writelines(comment_content.split('[/QUOTE]')[1] + '\n')
                                            else:
                                                data_export.writelines(comment_content + '\n\n')

        i = i + 1
data_export.close()
data_source.close()

