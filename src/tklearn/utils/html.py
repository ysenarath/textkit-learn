from dataclasses import fields, is_dataclass
from typing import Any, Mapping

__all__ = [
    "repr_html",
]

css = """
    <style>
    .attributes {
        display: flex;
        flex-wrap: wrap;
    }
    .attribute-pair {
        border-radius: 20px;
        padding: 5px 0;
        margin: 5px;
        box-sizing: border-box;
        display: flex;
        overflow: hidden;
    }
    .attribute-name {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 5px 10px;
        flex-grow: 1;
        text-align: center;
        background-color: #f0f0f0;
    }
    .attribute-value {
        padding: 5px 10px;
        flex-grow: 1;
        text-align: center;
        background-color: #f9f9f9;
    }
    .card {
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
    }
    .header {
        font-weight: bold;
        border-bottom: 1px solid #ccc;
        margin-left: -10px;
        margin-right: -10px;
        padding-left: 10px;
        padding-right: 10px;
        padding-bottom: 10px;
        margin-bottom: 10px;
        text-transform: capitalize;
        color: #333;
    }
    .iterable {
        display: flex;
        flex-wrap: wrap;
    }
    .iterable-item {
        display: inline-block;
    }
    .iterable-item:hover {
        background-color: #e0e0e0;
        border-color: #b0b0b0;
    }
    .mapping {
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .invisible {
        color: transparent;
        width: 0;
        padidng: 0;
        margin: 0;
    }
    .hidden {
        display: none;
    }
    .button {
        background-color: #ddd; /* Light gray */
        border: none;
        color: #333; /* Dark gray */
        text-align: center;
        text-decoration: none;
        display: inline-block;
        cursor: pointer;
        font-size: 0.8em; /* Smaller font size */
        padding: 5px;
        min-width: 10px;
        border-radius: 5px;
    }
    .button-group {
        display: flex;
        justify-content: center;
    }
    .button-group > .button:first-child {
        border-radius: 5px 0 0 5px;
    }
    .button-group > .button:last-child {
        border-radius: 0 5px 5px 0;
    }
    </style>
"""

js = """
<script>
document.querySelectorAll('.show-more-button').forEach((button) => {
    let pressTimer;
    let countdown;

    button.addEventListener('mouseup', (event) => {
        clearTimeout(pressTimer);
        clearInterval(countdown);
        button.textContent = '→';  // reset button text
        const buttonGroup = button.parentElement;
        const iterable = buttonGroup.parentElement;
        const hiddenItems = Array.from(iterable.querySelectorAll('.iterable-item.hidden'));
        for (let i = 0; i < Math.min(5, hiddenItems.length); i++) {
            hiddenItems[i].classList.remove('hidden');
        }
        if (hiddenItems.length <= 5) {
            button.style.display = 'none';
        }
        if (hiddenItems.length > 0) {
            buttonGroup.querySelector('.show-less-button').style.display = 'inline-block';
        }
    });

    button.addEventListener('mousedown', (event) => {
        clearTimeout(pressTimer);
        clearInterval(countdown);
        let countdownValue = 3;  // countdown from 3
        button.textContent = countdownValue;
        countdown = window.setInterval(() => {
            countdownValue--;
            if (countdownValue >= 0) {
                button.textContent = countdownValue;
            } else {
                window.clearInterval(countdown);
            }
        }, 1000);  // update every 1s
        pressTimer = window.setTimeout(() => {
            const buttonGroup = button.parentElement;
            const iterable = buttonGroup.parentElement;
            const hiddenItems = Array.from(iterable.querySelectorAll('.iterable-item.hidden'));
            for (let i = 0; i < hiddenItems.length; i++) {
                hiddenItems[i].classList.remove('hidden');
            }
            button.style.display = 'none';
            buttonGroup.querySelector('.show-less-button').style.display = 'inline-block';
        }, 3000);  // long press for 3s
    });
});

document.querySelectorAll('.show-less-button').forEach((button) => {
    button.addEventListener('click', (event) => {
        const buttonGroup = button.parentElement;
        const iterable = buttonGroup.parentElement;
        const shownItems = Array.from(iterable.querySelectorAll('.iterable-item:not(.hidden)'));
        for (let i = 5; i < shownItems.length; i++) {
            shownItems[i].classList.add('hidden');
        }
        button.style.display = 'none';
        const showMoreButton = buttonGroup.querySelector('.show-more-button');
        showMoreButton.textContent = '→';
        if (shownItems.length > 5) {
            showMoreButton.style.display = 'inline-block';
        } else {
            showMoreButton.style.display = 'none';
        }
    });
});
</script>
"""


def repr_html(obj) -> str:
    return css + repr_html_any(obj) + js


def repr_html_any(value: Any) -> str:
    if is_dataclass(value):
        return repr_html_dataclass(value)
    if isinstance(value, (list, tuple, set)):
        return repr_html_iterable(value)
    if isinstance(value, Mapping):
        return repr_html_mapping(value)
    else:
        return (
            repr(value)
            .replace("'", "<span class='invisible'>'</span>")
            .replace('"', '<span class="invisible">"</span>')
        )


def repr_html_dataclass(obj) -> str:
    fields_list = list(fields(obj))
    fields_html = "".join(
        f"<div class='attribute-pair'><span class='attribute-name'>{fields_list[i].name}:</span><span class='attribute-value'>{repr_html_any(getattr(obj, fields_list[i].name))}</span></div>"
        for i in range(len(fields_list))
    )
    return f"<div class='card'><div class='header'>{obj.__class__.__name__}</div><div class='attributes'>{fields_html}</div></div>"


def repr_html_iterable(obj, limit=5) -> str:
    items_html = "".join(
        f"<div class='iterable-item{' hidden' if index >= limit else ''}'>{repr_html_any(item)}</div>"
        for index, item in enumerate(obj)
    )
    show_more_button = f"""
    <button class='button show-more-button' style='display: {('none' if len(obj) <= limit else 'inline-block')}'>→</button>
    """
    show_less_button = """
    <button class='button show-less-button' style='display: none'>←</button>
    """
    button_group = (
        "<div class='button-group'>" + show_less_button + show_more_button + "</div>"
    )
    return f"<div class='iterable'>{items_html}{button_group}</div>"


def repr_html_mapping(obj: Mapping) -> str:
    items_html = "".join(
        f"<div class='mapping-item'><span class='mapping-item-key'>{repr_html_any(k)}:</span><span class='mapping-item-value'>{repr_html_any(v)}</span></div>"
        for k, v in obj.items()
    )
    return f"<div class='mapping'>{items_html}</div>"


def repr_html_pipeline(obj) -> str:
    return f"<div class='card'><div class='header'>{obj.__class__.__name__}</div><div class='attributes'><div class='attribute-pair'><span class='attribute-name'>steps:</span><span class='attribute-value'>{repr_html_iterable(obj.steps)}</span></div></div></div>"


def repr_repr_union(obj) -> str:
    return f"<div class='card'><div class='header'>{obj.__class__.__name__}</div><div class='attributes'><div class='attribute-pair'><span class='attribute-name'>types:</span><span class='attribute-value'>{repr_html_any(obj.steps)}</span></div></div></div>"
