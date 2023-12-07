# ObyGaze12 dataset for visual objectification task

The ObyGaze dataset provides dense annotations of levels and concepts of objectification on 12 films from the [MovieGraphs dataset](https://paperswithcode.com/dataset/moviegraphs). The movie clips correspond to the clips in MovieGraphs.

## Column names

- **idx**: the clip ID
- **util**: whether the clip has MovieGraphs annotations
- **clip**: the clip start and end timestamp
- **label**: the level of objectification Easy Negative, Hard Negative, Not Sure, or Sure
- **overlap ratio**: the ratio of overlap between the expert annotation and MovieGraphs clip
- **concepts**: the list of objectification concepts annotated in the case that the label is not Easy Negative
- **id**: clip ID with IMDB film ID
- **movie**: IMDB ID of film
- **srt_name**: subtitle file from MovieGraphs
- **graph_number**: corresponding graph ID in MovieGraphs. -1 indicates that the graph is not available

