"""Command-line interface for Home Person ID system."""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click

from src.config import load_config
from src.database.repository import Repository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_repository(config_path: str) -> Repository:
    """Get database repository from config."""
    config = load_config(config_path)
    return Repository(config.database.path)


@click.group()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx, config):
    """Home Person ID - Person identification system for home environments."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# ==================== Person Management ====================


def _add_faces_from_images(repo, person_id, image_patterns, recognizer):
    """Helper to add faces from image files with glob support."""
    import cv2
    import glob

    added_count = 0
    for pattern in image_patterns:
        # Expand glob patterns
        paths = glob.glob(pattern)
        if not paths:
            # Not a glob, treat as literal path
            paths = [pattern]

        for image_path in paths:
            path = Path(image_path)
            if not path.exists():
                click.echo(f"Warning: Image not found: {image_path}", err=True)
                continue

            try:
                image = cv2.imread(str(path))
                if image is None:
                    click.echo(f"Warning: Could not read image: {image_path}", err=True)
                    continue

                click.echo(f"Processing: {path.name} ({image.shape[1]}x{image.shape[0]})")

                # Detect and extract face embedding
                result = recognizer.detect_faces(image)
                if result.faces:
                    for face in result.faces:
                        embedding = recognizer.extract_embedding(image, face)
                        if embedding is not None:
                            repo.add_face_embedding(person_id, embedding, str(path))
                            click.echo(f"  Added face: {face.width:.0f}x{face.height:.0f} conf={face.confidence:.2f}")
                            added_count += 1
                else:
                    click.echo(f"  No face detected (image may be too small or face not frontal)")

            except Exception as e:
                click.echo(f"Error processing {image_path}: {e}", err=True)

    return added_count


@cli.command("add-person")
@click.argument("name")
@click.option("--images", "-i", multiple=True, help="Path(s) to face images (supports wildcards)")
@click.option("--camera", help="Camera ID for live capture")
@click.option("--capture", type=int, default=5, help="Number of frames to capture")
@click.pass_context
def add_person(ctx, name, images, camera, capture):
    """Add a new person to the system.

    Examples:
        add-person "John Doe" -i "./photos/john/*.jpg"
        add-person "Jane Doe" --camera front_door --capture 5
    """
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)
    repo = Repository(config.database.path)

    # Create person
    person = repo.create_person(name)
    click.echo(f"Created person: {person.id} - {name}")

    if images:
        from src.recognition.face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer(config.face_recognition)
        added = _add_faces_from_images(repo, person.id, images, recognizer)
        click.echo(f"Added {added} face(s) for {name}")

    elif camera:
        # Live capture from camera
        click.echo(f"Live capture from camera '{camera}' - capturing {capture} frames")
        click.echo("Press 'c' to capture a frame, 'q' to quit")

        from src.recognition.face_recognizer import FaceRecognizer
        from src.stream.rtsp_client import RTSPClient

        cam_config = config.get_camera(camera)
        if not cam_config:
            click.echo(f"Error: Camera '{camera}' not found in config", err=True)
            return

        recognizer = FaceRecognizer(config.face_recognition)

        import cv2

        with RTSPClient(camera, cam_config.rtsp_url, target_fps=10, use_nvdec=cam_config.use_nvdec) as client:
            captured = 0
            while captured < capture:
                frame = client.get_frame(timeout=2.0)
                if frame is None:
                    continue

                # Show frame
                display = frame.image.copy()
                result = recognizer.detect_faces(frame.image)

                for face in result.faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    display,
                    f"Captured: {captured}/{capture} - Press 'c' to capture, 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Live Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    if result.faces:
                        face = result.faces[0]  # Use first face
                        embedding = recognizer.extract_embedding(frame.image, face)
                        if embedding is not None:
                            repo.add_face_embedding(person.id, embedding)
                            captured += 1
                            click.echo(f"Captured face {captured}/{capture}")
                    else:
                        click.echo("No face detected - try again")

            cv2.destroyAllWindows()

    embeddings = repo.get_face_embeddings(person.id)
    click.echo(f"Person '{name}' has {len(embeddings)} face embeddings")


@cli.command("add-faces")
@click.option("--person", "-p", required=True, help="Person name or ID")
@click.option("--images", "-i", multiple=True, help="Path(s) to face images (supports wildcards)")
@click.option("--camera", help="Camera ID to capture from")
@click.option("--capture", type=int, default=1, help="Number of frames to capture from camera")
@click.option("--interactive", is_flag=True, help="Interactive mode: press 'c' to capture, 'q' to quit")
@click.pass_context
def add_faces(ctx, person, images, camera, capture, interactive):
    """Add face images to an existing person.

    Examples:
        add-faces -p "John Doe" -i "./photos/john/*.jpg"
        add-faces -p "Ati" --camera living_room
        add-faces -p "Ati" --camera living_room --interactive
        add-faces -p 1 --camera living_room --capture 3
    """
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)
    repo = Repository(config.database.path)

    # Find person by ID or name
    try:
        person_id = int(person)
        person_obj = repo.get_person(person_id)
    except ValueError:
        # Search by name
        persons = repo.get_all_persons()
        person_obj = next((p for p in persons if p.name.lower() == person.lower()), None)

    if not person_obj:
        click.echo(f"Person '{person}' not found", err=True)
        return

    if not images and not camera:
        click.echo("Error: Either --images or --camera is required", err=True)
        return

    click.echo(f"Adding faces to: {person_obj.name} (ID: {person_obj.id})")

    from src.recognition.face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer(config.face_recognition)
    added = 0

    if images:
        added = _add_faces_from_images(repo, person_obj.id, images, recognizer)

    if camera:
        # Capture from camera
        import cv2
        from src.stream.rtsp_client import RTSPClient

        cam_config = config.get_camera(camera)
        if not cam_config:
            click.echo(f"Camera '{camera}' not found", err=True)
            return

        if interactive:
            click.echo(f"Interactive capture from {cam_config.name}")
            click.echo("Press 'c' to capture, 'q' to quit")

            with RTSPClient(camera, cam_config.rtsp_url, target_fps=cam_config.fps, use_nvdec=cam_config.use_nvdec) as client:
                while True:
                    frame = client.get_frame(timeout=2.0)
                    if frame is None:
                        continue

                    display = frame.image.copy()

                    # Detect faces and draw boxes
                    result = recognizer.detect_faces(frame.image)
                    for face in result.faces:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display, f"{face.width:.0f}x{face.height:.0f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Status
                    status = f"Faces: {result.count} | Added: {added} | Press 'c' to capture, 'q' to quit"
                    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("Add Faces", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        if result.faces:
                            face = result.faces[0]
                            if face.embedding is not None:
                                repo.add_face_embedding(person_obj.id, face.embedding, f"camera_{camera}")
                                click.echo(f"  Captured face: {face.width:.0f}x{face.height:.0f} conf={face.confidence:.2f}")
                                added += 1
                            else:
                                click.echo("  Face found but no embedding extracted")
                        else:
                            click.echo("  No face detected - try again")

                cv2.destroyAllWindows()
        else:
            # Automatic capture
            click.echo(f"Capturing {capture} frame(s) from {cam_config.name}...")

            with RTSPClient(camera, cam_config.rtsp_url, target_fps=cam_config.fps, use_nvdec=cam_config.use_nvdec) as client:
                captured = 0
                attempts = 0
                while captured < capture and attempts < capture * 10:
                    attempts += 1
                    frame = client.get_frame(timeout=2.0)
                    if frame is None:
                        continue

                    result = recognizer.detect_faces(frame.image)
                    if result.faces:
                        face = result.faces[0]
                        if face.embedding is not None:
                            repo.add_face_embedding(person_obj.id, face.embedding, f"camera_{camera}")
                            click.echo(f"  Captured face: {face.width:.0f}x{face.height:.0f} conf={face.confidence:.2f}")
                            added += 1
                            captured += 1
                        else:
                            click.echo(f"  Face found but no embedding extracted")
                    else:
                        click.echo(f"  No face in frame, retrying...")

            if captured < capture:
                click.echo(f"Warning: Only captured {captured}/{capture} faces")

    embeddings = repo.get_face_embeddings(person_obj.id)
    click.echo(f"Added {added} face(s). Total faces for {person_obj.name}: {len(embeddings)}")


@cli.command("list-persons")
@click.pass_context
def list_persons(ctx):
    """List all known persons."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    persons = repo.get_all_persons()

    if not persons:
        click.echo("No persons registered")
        return

    click.echo(f"{'ID':<6} {'Name':<30} {'Faces':<8} {'Created'}")
    click.echo("-" * 60)

    for person in persons:
        embeddings = repo.get_face_embeddings(person.id)
        click.echo(
            f"{person.id:<6} {person.name:<30} {len(embeddings):<8} {person.created_at.strftime('%Y-%m-%d')}"
        )


@cli.command("show-person")
@click.option("--id", "person_id", type=int, required=True, help="Person ID")
@click.pass_context
def show_person(ctx, person_id):
    """Show details for a person."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    person = repo.get_person(person_id)
    if not person:
        click.echo(f"Person {person_id} not found", err=True)
        return

    embeddings = repo.get_face_embeddings(person_id)
    events = repo.get_events(person_id=person_id, limit=10)

    click.echo(f"Person: {person.name} (ID: {person.id})")
    click.echo(f"Created: {person.created_at}")
    click.echo(f"Active: {person.is_active}")
    click.echo(f"Face embeddings: {len(embeddings)}")
    click.echo(f"\nRecent events ({len(events)}):")

    for event in events:
        click.echo(f"  {event.timestamp} - {event.event_type} on {event.camera_id}")


@cli.command("remove-person")
@click.option("--id", "person_id", type=int, required=True, help="Person ID")
@click.confirmation_option(prompt="Are you sure you want to remove this person?")
@click.pass_context
def remove_person(ctx, person_id):
    """Remove a person from the system."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    if repo.delete_person(person_id):
        click.echo(f"Person {person_id} removed")
    else:
        click.echo(f"Person {person_id} not found", err=True)


# ==================== Camera Operations ====================


@cli.command("test-camera")
@click.argument("camera_id")
@click.pass_context
def test_camera(ctx, camera_id):
    """Test connection to a camera."""
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)

    cam_config = config.get_camera(camera_id)
    if not cam_config:
        click.echo(f"Camera '{camera_id}' not found in config", err=True)
        return

    click.echo(f"Testing camera: {cam_config.name}")
    click.echo(f"URL: {cam_config.rtsp_url}")
    decoder = "NVDEC" if cam_config.use_nvdec else "FFmpeg"
    click.echo(f"Decoder: {decoder}")

    # Use RTSPClient to test connection (reuses NVDEC pipeline logic)
    client = RTSPClient(
        camera_id=camera_id,
        rtsp_url=cam_config.rtsp_url,
        target_fps=5,
        use_nvdec=cam_config.use_nvdec,
    )

    if not client._connect():
        click.echo("FAILED: Could not connect to camera", err=True)
        if cam_config.use_nvdec:
            click.echo("  NVDEC requires GStreamer and nvidia plugins")
        return

    ret, frame = client._cap.read()
    client._release_capture()

    if ret and frame is not None:
        h, w = frame.shape[:2]
        click.echo(f"SUCCESS: Connected - {w}x{h}")
    else:
        click.echo("FAILED: Could not read frame", err=True)


@cli.command("preview")
@click.option("--camera", required=True, help="Camera ID to preview")
@click.option("--show-zones", is_flag=True, help="Show exclusion zones and polygon zones")
@click.option("--scale", type=float, default=1.0, help="Scale factor for display (e.g., 0.5 for half size)")
@click.option("--reid", is_flag=True, help="Enable Re-ID (re-identify persons who leave and return)")
@click.option("--save-snapshots", is_flag=True, help="Save snapshots on face/Re-ID matches")
@click.pass_context
def preview(ctx, camera, show_zones, scale, reid, save_snapshots):
    """Preview camera with detections."""
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)

    cam_config = config.get_camera(camera)
    if not cam_config:
        click.echo(f"Camera '{camera}' not found", err=True)
        return

    from src.detection.person_detector import PersonDetector
    from src.stream.rtsp_client import RTSPClient
    from src.tracking.zone_manager import ZoneManager
    from src.preview import (
        CameraProcessor, IdentificationManager, TrackRenderer, ZoneRenderer,
        SnapshotSaver, draw_status_bar, draw_motion_indicator
    )

    import cv2

    # Initialize person detector (shared)
    click.echo("Warming up person detector...")
    detector = PersonDetector(
        model_path=config.detection.model,
        confidence_threshold=config.detection.confidence_threshold,
        nms_iou_threshold=config.detection.nms_iou_threshold,
        num_threads=config.detection.num_threads,
    )
    detector.warmup()

    # Initialize zone manager
    zone_manager = None
    zone_renderer = None
    if config.zones and config.zones.zones:
        zone_manager = ZoneManager(config.zones)
        zone_renderer = ZoneRenderer(zone_manager)
        click.echo(f"Loaded {len(config.zones.zones)} polygon zones")

    # Load face gallery
    face_gallery = []
    if config.face_recognition.enabled:
        repo = Repository(config.database.path)
        persons = repo.get_all_persons()
        for person in persons:
            embeddings = repo.get_face_embeddings(person.id)
            for emb_id, embedding in embeddings:
                face_gallery.append((person.id, person.name, embedding))
        click.echo(f"Loaded {len(face_gallery)} face embeddings for {len(persons)} persons")

    # Initialize identification manager
    id_manager = IdentificationManager(config, face_gallery)
    if reid or face_gallery:
        click.echo("Re-ID enabled" if reid else "Face recognition enabled")
    click.echo("Warming up Re-ID extractor...")
    id_manager.warmup()

    # Initialize track renderer
    track_renderer = TrackRenderer(zone_manager)

    # Initialize snapshot saver
    snapshot_saver = SnapshotSaver(config.snapshots.path, enabled=save_snapshots)
    if save_snapshots:
        click.echo(f"Snapshots will be saved to: {config.snapshots.path}")

    # Initialize camera processor
    cam_processor = CameraProcessor(
        camera_id=camera,
        config=config,
        person_detector=detector,
        id_manager=id_manager,
        exclusion_zones=cam_config.exclusion_zones,
    )

    click.echo(f"Preview: {cam_config.name} - Press 'q' to quit")

    # Create resizable window (keeps full resolution, window handles scaling)
    window_name = "Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with RTSPClient(camera, cam_config.rtsp_url, target_fps=cam_config.fps, use_nvdec=cam_config.use_nvdec) as client:
        window_size_set = False
        while True:
            frame_data = client.get_frame(timeout=2.0)
            if frame_data is None:
                continue

            frame = frame_data.image
            display = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # Draw exclusion zones
            if show_zones and cam_config.exclusion_zones:
                for zone in cam_config.exclusion_zones:
                    x1, y1 = int(zone.x1 * frame_w), int(zone.y1 * frame_h)
                    x2, y2 = int(zone.x2 * frame_w), int(zone.y2 * frame_h)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, "EXCLUDE", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw polygon zones
            if show_zones and zone_renderer:
                zone_renderer.draw_zones(display, camera)

            # Process frame
            result = cam_processor.process_frame(frame, log_callback=click.echo)

            # Save snapshots for match events
            for match_event in result.match_events:
                saved_path = snapshot_saver.save_match(match_event, frame)
                if saved_path:
                    click.echo(f"  Saved: {saved_path}")

            # Draw tracks
            for track in result.tracks:
                label, color = track_renderer.get_track_label_and_color(
                    track.track_id, track.identity, camera, track.bbox,
                    frame_w, frame_h, track.stationary_time, show_reid_score=reid
                )
                track_renderer.draw_track(display, track.bbox, label, color)

            # Status
            if result.has_motion or result.track_count > 0:
                status = f"Tracks: {result.track_count} | Identified: {result.identified_count} | Gallery: {id_manager.gallery_size}"
            else:
                status = "No motion"

            draw_status_bar(display, status)
            draw_motion_indicator(display, result.has_motion)

            # Set initial window size (once), but keep full resolution image
            if not window_size_set:
                cv2.resizeWindow(window_name, int(frame_w * scale), int(frame_h * scale))
                window_size_set = True

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Periodic cleanup
            id_manager.cleanup_expired()

    cv2.destroyAllWindows()


# ==================== Event Queries ====================


@cli.command("events")
@click.option("--last", default="24h", help="Time range (e.g., 1h, 24h, 7d)")
@click.option("--camera", help="Filter by camera ID")
@click.option("--type", "event_type", help="Filter by event type")
@click.option("--limit", default=50, help="Maximum events to show")
@click.pass_context
def events(ctx, last, camera, event_type, limit):
    """Query recent events."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    # Parse time range
    if last.endswith("h"):
        hours = int(last[:-1])
        since = datetime.utcnow() - timedelta(hours=hours)
    elif last.endswith("d"):
        days = int(last[:-1])
        since = datetime.utcnow() - timedelta(days=days)
    else:
        since = datetime.utcnow() - timedelta(hours=24)

    events = repo.get_events(
        camera_id=camera,
        event_type=event_type,
        since=since,
        limit=limit,
    )

    if not events:
        click.echo("No events found")
        return

    click.echo(f"{'Time':<20} {'Type':<20} {'Camera':<15} {'Person'}")
    click.echo("-" * 70)

    for event in events:
        person_name = f"ID:{event.person_id}" if event.person_id else "-"
        click.echo(
            f"{event.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{event.event_type:<20} {event.camera_id:<15} {person_name}"
        )


@cli.command("occupancy")
@click.pass_context
def occupancy(ctx):
    """Show current occupancy (who is home)."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    tracks = repo.get_active_tracks()

    identified = []
    unknown = 0

    for track in tracks:
        if track.person_id:
            person = repo.get_person(track.person_id)
            if person:
                identified.append(person.name)
        else:
            unknown += 1

    click.echo(f"Active tracks: {len(tracks)}")
    click.echo(f"Identified: {', '.join(identified) if identified else 'None'}")
    click.echo(f"Unknown: {unknown}")


@cli.command("tracks")
@click.option("--active", is_flag=True, help="Show only active tracks")
@click.pass_context
def tracks(ctx, active):
    """Show current tracks."""
    config_path = ctx.obj["config_path"]
    repo = get_repository(config_path)

    if active:
        tracks = repo.get_active_tracks()
    else:
        tracks = repo.get_recent_tracks(hours=1)

    if not tracks:
        click.echo("No tracks found")
        return

    click.echo(f"{'Track ID':<15} {'Person':<20} {'Camera':<15} {'Status':<10} {'Last Seen'}")
    click.echo("-" * 80)

    for track in tracks:
        person_name = "-"
        if track.person_id:
            person = repo.get_person(track.person_id)
            if person:
                person_name = person.name

        click.echo(
            f"{track.id:<15} {person_name:<20} {track.last_camera_id or '-':<15} "
            f"{track.status:<10} {track.last_seen.strftime('%H:%M:%S')}"
        )


@cli.command("draw-zones")
@click.option("--camera", required=True, help="Camera ID to draw zones on")
@click.option("--scale", type=float, default=1.0, help="Scale factor for display")
@click.pass_context
def draw_zones(ctx, camera, scale):
    """Interactively draw polygon zones on camera.

    Controls:
    - LEFT CLICK: Add polygon point
    - RIGHT CLICK: Complete current polygon
    - 'n': Start new zone (prompts for name)
    - 'u': Undo last point
    - 'd': Delete last completed zone
    - 'p': Print YAML output
    - 'q': Quit
    """
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)

    cam_config = config.get_camera(camera)
    if not cam_config:
        click.echo(f"Camera '{camera}' not found", err=True)
        return

    from src.stream.rtsp_client import RTSPClient
    import cv2
    import numpy as np

    # State for zone drawing
    zones = []  # List of (name, polygon_points)
    current_polygon = []  # Points being drawn
    current_zone_name = "zone_1"

    # Colors for zones (cycling through)
    zone_colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Store frame dimensions
    frame_w, frame_h = 0, 0
    display_scale = scale

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, zones, current_zone_name

        # Adjust for scale
        actual_x = int(x / display_scale)
        actual_y = int(y / display_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current polygon
            current_polygon.append((actual_x, actual_y))
            click.echo(f"Added point: ({actual_x}, {actual_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete current polygon
            if len(current_polygon) >= 3:
                zones.append((current_zone_name, current_polygon.copy()))
                click.echo(f"Completed zone '{current_zone_name}' with {len(current_polygon)} points")
                current_polygon = []
                # Auto-increment zone name
                current_zone_name = f"zone_{len(zones) + 1}"
            else:
                click.echo("Need at least 3 points to complete a polygon")

    click.echo(f"Drawing zones on camera: {cam_config.name}")
    click.echo("Controls:")
    click.echo("  LEFT CLICK: Add polygon point")
    click.echo("  RIGHT CLICK: Complete current polygon")
    click.echo("  'n': Start new zone (prompts for name)")
    click.echo("  'u': Undo last point")
    click.echo("  'd': Delete last completed zone")
    click.echo("  'p': Print YAML output")
    click.echo("  'q': Quit")

    cv2.namedWindow("Draw Zones")
    cv2.setMouseCallback("Draw Zones", mouse_callback)

    with RTSPClient(camera, cam_config.rtsp_url, target_fps=cam_config.fps, use_nvdec=cam_config.use_nvdec) as client:
        while True:
            frame = client.get_frame(timeout=2.0)
            if frame is None:
                continue

            display = frame.image.copy()
            frame_h, frame_w = display.shape[:2]

            # Draw completed zones (filled semi-transparent)
            for i, (zone_name, polygon) in enumerate(zones):
                color = zone_colors[i % len(zone_colors)]
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

                # Draw filled polygon with transparency
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

                # Draw polygon outline
                cv2.polylines(display, [pts], True, color, 2)

                # Draw zone name
                if polygon:
                    centroid_x = int(sum(p[0] for p in polygon) / len(polygon))
                    centroid_y = int(sum(p[1] for p in polygon) / len(polygon))
                    cv2.putText(display, zone_name, (centroid_x - 30, centroid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw current polygon being created (green dashed)
            if current_polygon:
                pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], False, (0, 255, 0), 2)

                # Draw points
                for px, py in current_polygon:
                    cv2.circle(display, (px, py), 5, (0, 255, 0), -1)

                # Draw line to close polygon (dashed preview)
                if len(current_polygon) >= 2:
                    cv2.line(display, current_polygon[-1], current_polygon[0],
                             (0, 255, 0), 1, cv2.LINE_AA)

            # Status bar
            status = f"Zone: {current_zone_name} | Points: {len(current_polygon)} | Completed: {len(zones)}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Scale display if needed
            if display_scale != 1.0:
                new_w = int(display.shape[1] * display_scale)
                new_h = int(display.shape[0] * display_scale)
                display = cv2.resize(display, (new_w, new_h))

            cv2.imshow("Draw Zones", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # New zone - prompt for name
                cv2.destroyWindow("Draw Zones")
                name = click.prompt("Enter zone name", default=current_zone_name)
                current_zone_name = name
                current_polygon = []
                cv2.namedWindow("Draw Zones")
                cv2.setMouseCallback("Draw Zones", mouse_callback)
                click.echo(f"Started new zone: {current_zone_name}")
            elif key == ord('u'):
                # Undo last point
                if current_polygon:
                    removed = current_polygon.pop()
                    click.echo(f"Removed point: {removed}")
            elif key == ord('d'):
                # Delete last completed zone
                if zones:
                    removed = zones.pop()
                    click.echo(f"Deleted zone: {removed[0]}")
            elif key == ord('p'):
                # Print YAML output
                click.echo("\n" + "=" * 50)
                click.echo("YAML Output (copy to config.yaml):")
                click.echo("=" * 50)
                click.echo("zones:")
                for zone_name, polygon in zones:
                    click.echo(f"  - name: \"{zone_name}\"")
                    click.echo("    cameras:")
                    click.echo(f"      {camera}:")
                    # Convert to normalized coordinates
                    norm_polygon = [
                        [round(p[0] / frame_w, 4), round(p[1] / frame_h, 4)]
                        for p in polygon
                    ]
                    click.echo(f"        polygon: {norm_polygon}")
                    click.echo("    max_handover_sec: 5.0")
                click.echo("=" * 50 + "\n")

    cv2.destroyAllWindows()


@cli.command("preview-multi")
@click.option("--cameras", "-c", multiple=True, help="Camera IDs to preview (default: all)")
@click.option("--scale", type=float, default=0.5, help="Scale factor for each camera display")
@click.option("--show-zones", is_flag=True, help="Show polygon zones")
@click.option("--save-snapshots", is_flag=True, help="Save snapshots on face/Re-ID matches and handovers")
@click.pass_context
def preview_multi(ctx, cameras, scale, show_zones, save_snapshots):
    """Preview multiple cameras with cross-camera handover tracking.

    Shows all cameras side-by-side with global track IDs, face recognition,
    Re-ID scores, and handover events.
    Press 'q' to quit.
    """
    config_path = ctx.obj["config_path"]
    config = load_config(config_path)

    import cv2
    import time
    import numpy as np
    from threading import Thread, Lock

    from src.detection.person_detector import PersonDetector
    from src.stream.rtsp_client import RTSPClient
    from src.tracking.global_tracker import GlobalTrackManager
    from src.tracking.zone_manager import ZoneManager
    from src.recognition.identity_linker import IdentityLinker
    from src.preview import (
        CameraProcessor, IdentificationManager, TrackRenderer, ZoneRenderer,
        SnapshotSaver, draw_motion_indicator
    )

    # Determine which cameras to use
    camera_ids = list(cameras) if cameras else [cam.id for cam in config.cameras]

    # Validate cameras
    cam_configs = {}
    for cam_id in camera_ids:
        cam_config = config.get_camera(cam_id)
        if not cam_config:
            click.echo(f"Camera '{cam_id}' not found", err=True)
            return
        cam_configs[cam_id] = cam_config

    click.echo(f"Starting multi-camera preview with {len(camera_ids)} cameras: {camera_ids}")

    # Initialize shared components
    repo = Repository(config.database.path)
    identity_linker = IdentityLinker(config.face_recognition, config.reid, repo)

    # Initialize global tracker with zones
    global_tracker = GlobalTrackManager(
        config.camera_topology, config.reid, identity_linker, repo,
        zones_config=config.zones,
    )

    # Initialize zone manager and renderer
    zone_manager = None
    zone_renderer = None
    if config.zones and config.zones.zones:
        zone_manager = ZoneManager(config.zones)
        zone_renderer = ZoneRenderer(zone_manager)
        click.echo(f"Loaded {len(config.zones.zones)} polygon zones")

    # Load face gallery
    face_gallery = []
    if config.face_recognition.enabled:
        persons = repo.get_all_persons()
        for person in persons:
            embeddings = repo.get_face_embeddings(person.id)
            for emb_id, embedding in embeddings:
                face_gallery.append((person.id, person.name, embedding))
        click.echo(f"Loaded {len(face_gallery)} face embeddings for {len(persons)} persons")

    # Initialize identification manager and track renderer
    id_manager = IdentificationManager(config, face_gallery)
    track_renderer = TrackRenderer(zone_manager)

    # Initialize snapshot saver
    snapshot_saver = SnapshotSaver(config.snapshots.path, enabled=save_snapshots)
    if save_snapshots:
        click.echo(f"Snapshots will be saved to: {config.snapshots.path}")

    # Initialize shared person detector
    click.echo("Warming up person detector...")
    person_detector = PersonDetector(
        model_path=config.detection.model,
        confidence_threshold=config.detection.confidence_threshold,
        nms_iou_threshold=config.detection.nms_iou_threshold,
        num_threads=config.detection.num_threads,
    )
    person_detector.warmup()

    click.echo("Warming up Re-ID extractor...")
    id_manager.warmup()

    # Helper functions for global track lookups
    def get_global_track_id(camera_id: str, local_track_id: int) -> str | None:
        global_track = global_tracker.get_global_track_for_local(camera_id, local_track_id)
        return global_track.track_id if global_track else None

    def get_cameras_seen(global_track_id: str) -> list[str] | None:
        for gt in global_tracker._tracks.values():
            if gt.track_id == global_track_id:
                return gt.cameras_seen
        return None

    # Initialize per-camera components
    frame_locks = {}
    latest_frames = {}
    cam_processors = {}
    last_processed_frames = {}  # Store last processed frames for handover snapshots
    last_track_bboxes = {}  # Store last bboxes for each track
    running = True

    for cam_id in camera_ids:
        frame_locks[cam_id] = Lock()
        latest_frames[cam_id] = None
        last_processed_frames[cam_id] = None
        cam_processors[cam_id] = CameraProcessor(
            camera_id=cam_id,
            config=config,
            person_detector=person_detector,
            id_manager=id_manager,
            exclusion_zones=cam_configs[cam_id].exclusion_zones,
            get_global_track_id=get_global_track_id,
            get_cameras_seen=get_cameras_seen,
        )

    # Frame capture threads
    def capture_frames(cam_id, cam_config):
        nonlocal running
        try:
            with RTSPClient(cam_id, cam_config.rtsp_url, target_fps=cam_config.fps, use_nvdec=cam_config.use_nvdec) as client:
                while running:
                    frame = client.get_frame(timeout=1.0)
                    if frame is not None:
                        with frame_locks[cam_id]:
                            latest_frames[cam_id] = frame
        except Exception as e:
            click.echo(f"Camera {cam_id} error: {e}", err=True)

    # Start capture threads
    for cam_id in camera_ids:
        Thread(target=capture_frames, args=(cam_id, cam_configs[cam_id]), daemon=True).start()

    click.echo("Cameras starting... Press 'q' to quit")
    click.echo("-" * 60)

    handover_log = []

    # Create resizable window
    window_name = "Multi-Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Main processing loop
    try:
        while running:
            displays = []
            current_time = time.time()
            current_frames = {}  # Store current frames for this iteration

            for cam_id in camera_ids:
                with frame_locks[cam_id]:
                    frame_data = latest_frames[cam_id]

                if frame_data is None:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"{cam_id}: Connecting...",
                                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    displays.append(placeholder)
                    continue

                frame = frame_data.image
                current_frames[cam_id] = frame.copy()
                display = frame.copy()
                frame_h, frame_w = frame.shape[:2]

                # Draw zones
                if show_zones and zone_renderer:
                    zone_renderer.draw_zones(display, cam_id)

                # Get last tracks for global tracker check
                cam_proc = cam_processors[cam_id]
                has_active = len(cam_proc.last_tracks) > 0 or global_tracker.has_active_tracks(cam_id)

                # Process frame (handles motion, detection, tracking, identification)
                result = cam_proc.process_frame(frame, log_callback=click.echo)

                # Save snapshots for match events
                for match_event in result.match_events:
                    saved_path = snapshot_saver.save_match(match_event, frame)
                    if saved_path:
                        click.echo(f"  Saved: {saved_path}")

                # Store track bboxes for handover snapshots
                for track in result.tracks:
                    last_track_bboxes[(cam_id, track.track_id)] = track.bbox

                # Global tracking (need to process after local tracking)
                if result.has_motion or has_active:
                    # Get the raw ByteTracker tracks for global tracker
                    local_tracks = cam_proc.last_tracks

                    global_result = global_tracker.process_local_tracks(
                        camera_id=cam_id,
                        local_tracks=local_tracks,
                        frame=frame,
                        new_track_ids=result.new_track_ids,
                        lost_track_ids=result.lost_track_ids,
                        has_motion=result.has_motion,
                    )

                    # Log and save handover events
                    for track_id, from_cam, to_cam in global_result.handovers_completed:
                        identity = id_manager.get_identity(track_id)
                        name = identity.person_name or "Unknown"
                        msg = f"[HANDOVER] {name} ({track_id}): {from_cam} -> {to_cam}"
                        click.echo(msg)
                        handover_log.append((current_time, msg))

                        # Save handover snapshot
                        from_frame = last_processed_frames.get(from_cam)
                        to_frame = current_frames.get(to_cam, frame)
                        if from_frame is not None:
                            from_bbox = last_track_bboxes.get((from_cam, track_id))
                            to_bbox = last_track_bboxes.get((to_cam, track_id))
                            saved_path = snapshot_saver.save_handover(
                                track_id=track_id,
                                person_name=name,
                                from_camera=from_cam,
                                to_camera=to_cam,
                                from_frame=from_frame,
                                to_frame=to_frame,
                                from_bbox=from_bbox,
                                to_bbox=to_bbox,
                            )
                            if saved_path:
                                click.echo(f"  Saved handover: {saved_path}")

                    for track_id in global_result.new_global_tracks:
                        click.echo(f"[NEW] {track_id} on {cam_id}")

                # Draw tracks
                for track in result.tracks:
                    label, color = track_renderer.get_track_label_and_color(
                        track.track_id, track.identity, cam_id, track.bbox,
                        frame_w, frame_h, track.stationary_time, track.cameras_seen
                    )
                    track_renderer.draw_track(display, track.bbox, label, color)

                # Camera label
                cv2.putText(display, f"{cam_configs[cam_id].name} | T:{result.track_count}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                draw_motion_indicator(display, result.has_motion)

                # Scale
                if scale != 1.0:
                    display = cv2.resize(display, (int(frame_w * scale), int(frame_h * scale)))

                displays.append(display)

                # Store frame for handover snapshots
                last_processed_frames[cam_id] = frame.copy()

            # Cleanup
            id_manager.cleanup_expired()

            # Arrange grid
            if displays:
                n = len(displays)
                cols = min(n, 2)
                rows = (n + cols - 1) // cols
                max_h = max(d.shape[0] for d in displays)
                max_w = max(d.shape[1] for d in displays)

                padded = []
                for d in displays:
                    if d.shape[0] < max_h or d.shape[1] < max_w:
                        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                        pad[:d.shape[0], :d.shape[1]] = d
                        padded.append(pad)
                    else:
                        padded.append(d)

                grid_rows = []
                for r in range(rows):
                    row = padded[r * cols:(r + 1) * cols]
                    while len(row) < cols:
                        row.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
                    grid_rows.append(np.hstack(row))
                grid = np.vstack(grid_rows)

                # Status bar
                pending = len(global_tracker._pending_handovers)
                identified = sum(1 for gid in id_manager._identities if id_manager.is_identified(gid))
                status = f"Identified: {identified} | Gallery: {id_manager.gallery_size} | Pending: {pending}"
                cv2.putText(grid, status, (10, grid.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow(window_name, grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    except KeyboardInterrupt:
        running = False

    running = False
    cv2.destroyAllWindows()

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("Session Summary:")
    click.echo("=" * 60)
    click.echo(f"Total handovers: {len(handover_log)}")
    if handover_log:
        click.echo("\nHandover Events:")
        for _, msg in handover_log:
            click.echo(f"  {msg}")


if __name__ == "__main__":
    cli()
