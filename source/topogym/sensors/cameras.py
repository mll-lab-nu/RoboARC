"""Camera helpers shared by datasets, live recordings and task adapters."""


def update_dynamic_camera(camera):
    if getattr(camera, "_attached_link", None) is not None:
        camera.move_to_attach()
    elif getattr(camera, "_followed_entity", None) is not None:
        camera.update_following()
