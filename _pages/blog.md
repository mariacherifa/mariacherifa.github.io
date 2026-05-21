---
layout: page
permalink: /blog/
title: Blog
description: My blog posts on Transformers and mathematics.
nav: true
nav_order: 7
---

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%B %d, %Y" }}

{{ post.description }}

---
{% endfor %}
