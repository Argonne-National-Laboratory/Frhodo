"""Offscreen render check for the Mechanism group box's layout."""
import pytest



pytestmark = pytest.mark.gui


def _assert_laid_out_inside(widget, group_box, name):
    """A widget must have real size and sit within the group box's rect."""
    assert widget.width() > 0 and widget.height() > 0, (
        f"{name} has zero size after layout: {widget.size()}"
    )
    assert widget.isVisibleTo(group_box), f"{name} is not visible within the group box"

    geometry = widget.geometry()
    assert group_box.rect().contains(geometry), (
        f"{name} geometry {geometry} falls outside the group box rect {group_box.rect()}"
    )


def test_mechanism_group_box_renders_thermo_and_transport_rows(main_window, tmp_path):
    main_window.show()
    main_window.app.processEvents()

    group_box = main_window.groupBox_12
    pixmap = group_box.grab()

    assert not pixmap.isNull(), "the Mechanism group box grabbed an empty pixmap"
    assert pixmap.width() > 0 and pixmap.height() > 0, (
        f"grabbed pixmap has zero size: {pixmap.size()}"
    )

    grab_path = tmp_path / "mechanism_group_box.png"
    pixmap.save(str(grab_path))
    assert grab_path.stat().st_size > 0, "the saved grab is an empty file"

    _assert_laid_out_inside(main_window.use_thermo_file_box, group_box, "use_thermo_file_box")
    _assert_laid_out_inside(
        main_window.thermo_select_comboBox, group_box, "thermo_select_comboBox"
    )
    _assert_laid_out_inside(
        main_window.use_transport_file_box, group_box, "use_transport_file_box"
    )
    _assert_laid_out_inside(
        main_window.transport_select_comboBox, group_box, "transport_select_comboBox"
    )
