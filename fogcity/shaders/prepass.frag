#version 410 core

flat in uint v_building_id;

layout(location = 0) out uint out_building_id;

void main()
{
    out_building_id = v_building_id;
}

