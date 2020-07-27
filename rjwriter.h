#pragma once

#include <rapidjson/writer.h>

struct RJWriter : rapidjson::Writer<rapidjson::StringBuffer>
{
    using rapidjson::Writer<rapidjson::StringBuffer>::Writer;
};
