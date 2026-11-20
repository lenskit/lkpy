# Summary of ml-100k

## Entities

- item (1.68 k, 1 attributes)
- user (943, 4 attributes)

### Entity `item`

- 1,682 instances

#### Attributes

| Name  | Layout |  Type  | Dimension | Count |     Size |
| :-----| :----: | :----: |---------: |-----: |--------: |
| title | scalar | string |         - | 1,682 | 45.1 KiB |

#### Schema

```
item_id: int32
title: string
```

### Entity `user`

- 943 instances

#### Attributes

| Name       | Layout |  Type  | Dimension | Count |     Size |
| :----------| :----: | :----: |---------: |-----: |--------: |
| age        | scalar | int64  |         - |   943 |  7.4 KiB |
| gender     | scalar | string |         - |   943 |  4.6 KiB |
| occupation | scalar | string |         - |   943 | 11.2 KiB |
| zip_code   | scalar | string |         - |   943 |  8.3 KiB |

#### Schema

```
user_id: int32
age: int64
gender: string
occupation: string
zip_code: string
```

## Relationships

- rating (100 k, 2 attributes, interaction)

### Relationship `rating`

- 100,000 records

#### Entities

| Name | Class | Unique Count |
| :----| :---: |------------: |
| user |  user |          943 |
| item |  item |        1,682 |

#### Attributes

| Name      | Layout |      Type     | Dimension |   Count |      Size |
| :---------| :----: | :-----------: |---------: |-------: |---------: |
| rating    | scalar |     float     |         - | 100,000 | 390.6 KiB |
| timestamp | scalar | timestamp[ns] |         - | 100,000 | 781.2 KiB |

## Data Tables

| Name   |    Rows |    Bytes |
| :------|-------: |--------: |
| item   |   1,682 | 51.7 KiB |
| rating | 100,000 |  1.9 MiB |
| user   |     943 | 35.1 KiB |
