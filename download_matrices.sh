#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p data

# Array of URLs
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/662_bus.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Geer.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Bodendiek/CurlCurl_4.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Goodwin/Goodwin_127.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/VanVelzen/Zd_Jac3_db.tar.gz"
)

# Loop through the URLs
for url in "${urls[@]}"; do
    echo "Processing $url"
    
    # Extract the filename from the URL
    filename=$(basename "$url")
    
    # Download the file into the data directory
    wget -P ./data "$url"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Downloaded $filename successfully."
        
        # Unpack the tar.gz file into the data directory
        # The --strip-components=1 option is often useful if the archive contains a single top-level directory
        # Adjust if the archive structure is different
        tar -xzf "./data/$filename" -C ./data
        
        if [ $? -eq 0 ]; then
            echo "Unpacked $filename successfully."
            
            # Remove the downloaded tar.gz file
            rm "./data/$filename"
            echo "Removed $filename."
        else
            echo "Error unpacking $filename."
        fi
    else
        echo "Error downloading $filename."
    fi
    echo "----------------------------------------"
done

echo "All matrices processed."