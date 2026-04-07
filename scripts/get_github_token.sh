#!/usr/bin/env bash

# GitHub device flow for CLI login

# Replace with your GitHub OAuth App Client ID
CLIENT_ID="YOUR_GITHUB_OAUTH_CLIENT_ID"
SCOPE="read:user models:read"

DEVICE_CODE_URL="https://github.com/login/device/code"
ACCESS_TOKEN_URL="https://github.com/login/oauth/access_token"

echo "Requesting device code from GitHub..."

# Request device code
RESPONSE=$(curl -s -X POST "$DEVICE_CODE_URL" \
  -H "Accept: application/json" \
  -d "client_id=$CLIENT_ID" \
  -d "scope=$SCOPE")

USER_CODE=$(echo "$RESPONSE" | grep -o '"user_code":"[^"]*' | cut -d'"' -f4)
VERIFICATION_URI=$(echo "$RESPONSE" | grep -o '"verification_uri":"[^"]*' | cut -d'"' -f4)
DEVICE_CODE=$(echo "$RESPONSE" | grep -o '"device_code":"[^"]*' | cut -d'"' -f4)
INTERVAL=$(echo "$RESPONSE" | grep -o '"interval":[0-9]*' | cut -d':' -f2)

if [ -z "$USER_CODE" ] || [ -z "$DEVICE_CODE" ]; then
    echo "Failed to get device code. Response:"
    echo "$RESPONSE"
    exit 1
fi

if [ -z "$INTERVAL" ]; then
    INTERVAL=5
fi

echo ""
echo "======================================================"
echo "Please visit: $VERIFICATION_URI"
echo "And enter the code: $USER_CODE"
echo "======================================================"
echo ""
echo "Waiting for authorization..."

# Poll for access token
while true; do
    sleep $INTERVAL
    
    TOKEN_RESPONSE=$(curl -s -X POST "$ACCESS_TOKEN_URL" \
      -H "Accept: application/json" \
      -d "client_id=$CLIENT_ID" \
      -d "device_code=$DEVICE_CODE" \
      -d "grant_type=urn:ietf:params:oauth:grant-type:device_code")
    
    ERROR=$(echo "$TOKEN_RESPONSE" | grep -o '"error":"[^"]*' | cut -d'"' -f4)
    
    if [ "$ERROR" = "authorization_pending" ]; then
        continue
    elif [ "$ERROR" = "slow_down" ]; then
        INTERVAL=$((INTERVAL + 5))
        continue
    elif [ -n "$ERROR" ]; then
        echo "Error: $ERROR"
        echo "$TOKEN_RESPONSE"
        exit 1
    else
        ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
        if [ -n "$ACCESS_TOKEN" ]; then
            echo ""
            echo "Successfully authenticated!"
            echo "Your access token is:"
            echo "$ACCESS_TOKEN"
            exit 0
        else
            echo "Failed to parse access token. Response:"
            echo "$TOKEN_RESPONSE"
            exit 1
        fi
    fi
done
