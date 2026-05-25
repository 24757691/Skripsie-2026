#!/usr/bin/env python3
"""
YouTube Channel Scraper for South African Languages
Hardcoded API key - run with language and max-videos
Filters for full news broadcasts only
"""

import os
import sys
import time
import re
import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ======================================================
# HARDCODED API KEY (DO NOT SHARE)
# ======================================================
YOUR_API_KEY = "AIzaSyCnw9ZBuJzEg9OPA_ZHsKZynZFURkUEMAc"
# ======================================================

# Channel IDs for different languages
CHANNELS = {
    'afrikaans': {
        'id': 'UCPde3flccFt8JwCQwmOCWsg',
        'name': 'SABC Nuus (Afrikaans)',
        'output_file': 'afrikaans_links.txt',
        'title_patterns': [
            r'Afrikaans Nuus \| \d{1,2} \w+ \d{4}',      # "Afrikaans Nuus | 31 Maart 2026"
            r'Afrikaanse Nuus \| \d{1,2} \w+ \d{4}',     # "Afrikaanse Nuus | 31 Maart 2026"
            r'SABC Nuus \| \d{1,2} \w+ \d{4}',           # "SABC Nuus | 31 Maart 2026"
        ]
    },
    'isixhosa': {
        'id': 'UCSO8qij295DSxMiofTLk9QQ',
        'name': 'SABC Iindaba (isiXhosa)',
        'output_file': 'isixhosa_links.txt',
        'title_patterns': [
            r'Iindaba zesiXhosa @\d{2}H\d{2} \| \d{1,2} \w+ \d{4}',   # "Iindaba zesiXhosa @13H00 | 31 March 2026"
            r'Iindaba zesiXhosa \| \d{1,2} \w+ \d{4}',                 # "Iindaba zesiXhosa | 31 March 2026"
            r'SABC Iindaba \| \d{1,2} \w+ \d{4}',                      # "SABC Iindaba | 31 March 2026"
        ]
    }
}

def get_channel_videos(youtube, channel_id, max_videos, language, title_patterns):
    """Get video links from a YouTube channel, filtered by title patterns"""
    video_urls = []
    next_page_token = None
    filtered_count = 0
    checked_count = 0
    
    print(f"\nFetching video links from channel...")
    if title_patterns:
        print(f"  Filtering for news broadcasts only")
        print(f"  Patterns: {' OR '.join([p[:30]+'...' for p in title_patterns[:2]])}")
    else:
        print(f"  No filters applied - fetching all videos")
    
    # First, get the uploads playlist ID from the channel
    try:
        channel_request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        channel_response = channel_request.execute()
        
        if not channel_response['items']:
            print("  ❌ Channel not found!")
            return []
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        print(f"  ✓ Found uploads playlist")
        
    except HttpError as e:
        print(f"  ❌ Error getting channel info: {e}")
        return []
    
    # Get videos from the uploads playlist, filtering by title
    while len(video_urls) < max_videos:
        try:
            playlist_request = youtube.playlistItems().list(
                part='snippet,contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            playlist_response = playlist_request.execute()
            
            for item in playlist_response['items']:
                title = item['snippet']['title']
                video_id = item['contentDetails']['videoId']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                checked_count += 1
                
                # Check if title matches any pattern
                is_match = False
                if title_patterns:
                    for pattern in title_patterns:
                        if re.search(pattern, title, re.IGNORECASE):
                            is_match = True
                            break
                else:
                    is_match = True  # No filters = match everything
                
                if is_match:
                    video_urls.append(video_url)
                    print(f"  ✓ [{len(video_urls)}] {title[:70]}...")
                    if len(video_urls) >= max_videos:
                        break
                else:
                    filtered_count += 1
                    # Print progress every 50 filtered videos
                    if filtered_count % 50 == 0:
                        print(f"  ⏭ Filtered {filtered_count} non-news videos (checked {checked_count})...")
            
            print(f"  Found {len(video_urls)} matching videos so far...")
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
                
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except HttpError as e:
            print(f"  ❌ Error fetching videos: {e}")
            break
    
    print(f"\n  ✓ Found {len(video_urls)} news broadcasts (filtered {filtered_count} non-news)")
    return video_urls

def save_links_to_file(video_urls, filename):
    """Save video links to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for url in video_urls:
                f.write(url + '\n')
        return True
    except Exception as e:
        print(f"  ❌ Error saving file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Scrape video links from YouTube channels')
    parser.add_argument('language', choices=['afrikaans', 'isixhosa'],
                       help='Language to scrape')
    parser.add_argument('--max-videos', '-m', type=int, default=500,
                       help='Maximum number of videos to fetch (default: 500)')
    
    args = parser.parse_args()
    
    language = args.language
    max_videos = args.max_videos
    
    # Get channel info for selected language
    channel_info = CHANNELS[language]
    output_file = channel_info['output_file']
    title_patterns = channel_info.get('title_patterns', [])
    
    print("\n" + "="*70)
    print(f"🎬 YOUTUBE CHANNEL SCRAPER")
    print("="*70)
    print(f"\nLanguage:    {language.upper()}")
    print(f"Channel:     {channel_info['name']}")
    print(f"Channel ID:  {channel_info['id']}")
    print(f"Output file: {output_file}")
    print(f"Max videos:  {max_videos}")
    
    # Connect to YouTube API
    print(f"\n[1/4] Connecting to YouTube API...")
    try:
        youtube = build('youtube', 'v3', developerKey=YOUR_API_KEY)
        print(f"  ✓ API connection successful")
    except Exception as e:
        print(f"  ❌ API connection failed: {e}")
        return
    
    # Fetch videos with filtering
    print(f"\n[2/4] Fetching videos from {channel_info['name']}...")
    video_urls = get_channel_videos(youtube, channel_info['id'], max_videos, language, title_patterns)
    
    if not video_urls:
        print(f"\n❌ No videos found!")
        return
    
    print(f"\n[3/4] Found {len(video_urls)} videos")
    
    # Save to file
    print(f"\n[4/4] Saving links to {output_file}...")
    if save_links_to_file(video_urls, output_file):
        print(f"  ✓ Successfully saved {len(video_urls)} links")
    else:
        print(f"  ❌ Failed to save file")
        return
    
    # Summary
    print("\n" + "="*70)
    print("✅ SCRAPING COMPLETE!")
    print("="*70)
    print(f"\n📊 SUMMARY:")
    print(f"  Language:      {language.upper()}")
    print(f"  Channel:       {channel_info['name']}")
    print(f"  Videos found:  {len(video_urls)}")
    print(f"  Output file:   {output_file}")
    
    # Show first 5 links as sample
    print(f"\n🎯 First 5 video links:")
    for i, url in enumerate(video_urls[:5], 1):
        print(f"  {i}. {url}")
    if len(video_urls) > 5:
        print(f"  ... and {len(video_urls) - 5} more")

if __name__ == "__main__":
    main()
