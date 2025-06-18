#Python Client for Segmind APIs

A comprehensive Python client for interacting with Segmind's AI models and APIs. This package provides easy access to various generative AI models offered by Segmind, including text-to-image, image-to-image, background removal, face enhancement, and more.

## Installation

```bash
pip install -e .
```


The Segmind API implements rate limiting to ensure fair usage. If you encounter a `429 Too Many Requests` error, it means you've exceeded the allowed number of requests in a given time period. Here are some strategies to handle rate limits:

1. **Implement exponential backoff**: When you receive a 429 error, wait for a short period before retrying, and increase the wait time exponentially with each retry.

2. **Check your API quota**: Monitor your remaining credits by checking the `x-remaining-credits` header in API responses.

3. **Optimize your requests**: Batch operations when possible and avoid making unnecessary API calls.

4. **Upgrade your plan**: If you consistently hit rate limits, consider upgrading to a higher tier plan with increased limits.

Example implementation of exponential backoff:

```python
import time
import random

def call_with_backoff(func, max_retries=5, initial_delay=1):
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and retries < max_retries:
                # Add jitter to avoid thundering herd problem
                wait_time = delay + random.uniform(0, 0.1) * delay
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")
```

## Credits

This package is a client for the Segmind API. For more information about Segmind and their services, visit [https://www.segmind.com/](https://www.segmind.com/).
