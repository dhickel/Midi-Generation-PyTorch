import os
import requests
from bs4 import BeautifulSoup

# Used to scrape midi files

# The URL of the webpage you want to scrape
url = "<url>"

# The directory where you want to save the files
save_path = "data/scraped/"

# Make directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Send a GET request to the URL
response = requests.get(url)

# If the GET request is successful, the status code will be 200
if response.status_code == 200:
    # Get the content of the response
    page_content = response.content

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(page_content, 'html.parser')

    # Find all the links in the HTML
    links = soup.find_all("a")

    for link in links:
        href = link.get("href")

        # If href is not None and is a .mid or .MID file
        if href and (href.endswith(".mid") or href.endswith(".MID")):
            # If href is a relative link, then concatenate it with the base url
            if not href.startswith("http"):
                href = url + href

            # Send a GET request to the URL of the .mid or .MID file
            file_response = requests.get(href)

            # If the GET request is successful, the status code will be 200
            if file_response.status_code == 200:
                # Get the content of the response
                file_content = file_response.content

                # Get the file name from the href
                file_name = href.split("/")[-1]

                # Open the file in write mode
                with open(os.path.join(save_path, file_name), "wb") as file:
                    # Write the content to the file
                    file.write(file_content)

print("Download completed.")
