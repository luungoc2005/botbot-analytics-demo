import json

def return_response(args, response):
    if args.callback_url.strip() != '':
        # from urllib import request, parse
        import requests

        print('Sending POST request to', args.callback_url)
        # data = json.dumps(response).encode('utf8')
        request_obj = requests.post(
            args.callback_url,
            data=json.dumps(response)
        )
