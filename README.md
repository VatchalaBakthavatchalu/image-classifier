# Image classifier Server (Python)

Server that runs a REST API which reads a publicly hosted image, and returns a classification. Accuracy is expected if category of image exists within imagenet, and if sufficient information clarity of category is present in the picture

## Installation

make sure you have the necessary dependencies, listed within '''setup.py'''

## Usage

TODO: Write usage instructions

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

TODO: Write credits

## License

TODO: Write license


Setup:
	(1) install dependencies: keras, flask, requests

start server: 


example curl:
curl http://localhost:5000/img1 -d "data=https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png" -X PUT


Header and a Brief description (should match package.json)
Example (if applicable)
Motivation (if applicable)
API Documentation: This will likely vary considerably from library to library.
Installation