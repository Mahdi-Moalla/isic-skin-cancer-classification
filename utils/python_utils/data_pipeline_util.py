"""
create data processing pipeline defined from json file
"""
import sys
import json
import importlib

def dynamic_import(module,
                   name,
                   attrs=None):
    """
    import by name
    """
    module = importlib.import_module(module)
    name=getattr(module, name)
    if attrs is not None:
        attrs=attrs.split('.')
        for comp in attrs:
            name=getattr(name, comp)
    return name


def create_pipeline(json_file_path):
    """
    create pipeline from json file
    """

    with open(json_file_path,
              "r",
              encoding="utf-8") as f_json:
        json_data=json.load(f_json)

    #print(json_data)

    transforms=[]

    import_prefix=json_data["import_prefix"]

    for transform_desc in json_data["transforms"]:
        transform_name=transform_desc["transform"]
        try:
            transform  =  dynamic_import(
                import_prefix,
                transform_name)
        except ValueError:
            print("unknown transform")
            sys.exit(1)
        args=transform_desc["args"]
        for k,v in args.items():
            if isinstance(v,dict) and list(v.keys())==["__import__"]:
                args[k]=dynamic_import(**v["__import__"])
        transforms.append(transform(**args))

    return dynamic_import(
        import_prefix,
        json_data["pipeline_composer"])(transforms)
