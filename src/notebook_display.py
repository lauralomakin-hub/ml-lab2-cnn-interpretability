from IPython.utils import capture
from IPython.display import HTML, display
import html

from src.pipeline_utils import run_example


def capture_example(image_path, title, target_layer="layer4"):
    with capture.capture_output() as cap:
        info = run_example(
            image_path,
            target_layer=target_layer
        )

    image_png = None

    for output in cap.outputs:
        if hasattr(output, "data") and "image/png" in output.data:
            image_png = output.data["image/png"]
            if isinstance(image_png, list):
                image_png = "".join(image_png)
            break

    if image_png is None:
        raise ValueError("No image output was captured from run_example().")

    return {
        "title": title,
        "image_png": image_png,
        "info": info,
        "stdout": cap.stdout,
    }


def show_examples_side_by_side(examples, target_layer="layer4"):
    results = []

    for image_path, title in examples:
        result = capture_example(
            image_path=image_path,
            title=title,
            target_layer=target_layer
        )
        results.append(result)

    blocks = []

    for result in results:
        info = result["info"]

        block = f"""
        <div style="flex: 1; padding: 8px; min-width: 250px;">
            <h4 style="text-align: center;">{html.escape(result["title"])}</h4>
            <img src="data:image/png;base64,{result["image_png"]}" style="width: 100%;">
            <p style="font-size: 14px;">
                <b>Prediction:</b> {html.escape(info["class_name"])}<br>
                <b>Confidence:</b> {info["confidence"]:.2%}<br>
                <b>Class index:</b> {info["class_index"]}<br>
                <b>Class id:</b> {html.escape(info["class_id"])}
            </p>

            <pre style="font-size: 12px; white-space: pre-wrap;">
{html.escape(result["stdout"])}
            </pre>
        </div>
        """

        blocks.append(block)

    display(HTML(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start;">
        {''.join(blocks)}
    </div>
    """))

    return [result["info"] for result in results]


def show_layers_side_by_side(image_path, layers=("layer1", "layer2", "layer3", "layer4")):
    results = []

    for layer in layers:
        result = capture_example(
            image_path=image_path,
            title=layer,
            target_layer=layer
        )
        results.append(result)

    blocks = []

    for result in results:
        info = result["info"]

        block = f"""
        <div style="flex: 1; padding: 8px; min-width: 220px;">
            <h4 style="text-align: center;">{html.escape(result["title"])}</h4>
            <img src="data:image/png;base64,{result["image_png"]}" style="width: 100%;">
            <p style="font-size: 14px;">
                <b>Prediction:</b> {html.escape(info["class_name"])}<br>
                <b>Confidence:</b> {info["confidence"]:.2%}<br>
                <b>Class index:</b> {info["class_index"]}<br>
                <b>Class id:</b> {html.escape(info["class_id"])}
            </p>
        </div>
        """

        blocks.append(block)

    display(HTML(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start;">
        {''.join(blocks)}
    </div>
    """))

    return [result["info"] for result in results]