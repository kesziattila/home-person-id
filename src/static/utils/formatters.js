// Server returns UTC times without 'Z' suffix — parse them as UTC so the
// browser converts to the viewer's local time. ALWAYS use this (or the
// formatters below) instead of `new Date(ts)`: naive parsing treats DB
// timestamps as local time, displaying raw UTC values.
export function parseUtcDate(isoString) {
    let utcString = isoString;
    if (!isoString.endsWith('Z') && !isoString.includes('+') && !isoString.includes('-', 10)) {
        utcString = isoString + 'Z';
    }
    return new Date(utcString);
}

// Absolute timestamp rendered in the viewer's local timezone.
export function formatDateTime(isoString) {
    if (!isoString) return '';
    return parseUtcDate(isoString).toLocaleString();
}

export function formatRelativeTime(isoString) {
    if (!isoString) return '';

    const date = parseUtcDate(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);

    if (diffSec < 0) return 'just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) {
        const mins = Math.floor(diffSec / 60);
        const secs = diffSec % 60;
        return secs > 0 ? `${mins}m ${secs}s ago` : `${mins}m ago`;
    }
    if (diffSec < 86400) {
        const hours = Math.floor(diffSec / 3600);
        const mins = Math.floor((diffSec % 3600) / 60);
        return mins > 0 ? `${hours}h ${mins}m ago` : `${hours}h ago`;
    }
    return `${Math.floor(diffSec / 86400)}d ago`;
}

// Runner-up person this close to a face makes it a risky gallery image: on 2026-10-19
// Judit faces with Virag >= ~0.2 made one more Virag face look like Judit offline
export const RUNNER_UP_WARN = 0.2;

/**
 * The two closest persons of a face suggestion in the current gallery (API match_scores),
 * as HTML: "Judit 0.47 · Virag 0.21" with a risky runner-up in yellow.
 */
export function formatMatchScores(scores) {
    if (!scores || scores.length === 0) return '';
    return scores.map((s, i) => {
        const risky = i > 0 && s.score >= RUNNER_UP_WARN;
        const cls = i === 0 ? 'text-gray-200' : (risky ? 'text-yellow-400' : 'text-gray-500');
        const title = i === 0
            ? 'Closest person in the current face gallery (max cosine): clearly highest but not near 1.0 = a new view of them'
            : (risky ? 'A second person is close: as a gallery image this face may also pull that person\'s faces'
                     : 'Second closest person: far = safe to enroll');
        return `<span class="${cls}" title="${title}">${s.person_name} ${s.score.toFixed(2)}</span>`;
    }).join('<span class="text-gray-600"> · </span>');
}

/**
 * Enrolment value badge of a face suggestion (API enrol_label / coverage, decided server-side:
 * routes/unidentified_faces.enrolment_label). Quality stays a separate badge.
 */
export function enrolBadge(face) {
    const scores = face.match_scores || [];
    const who = scores.length ? scores[0].person_name : 'its person';
    if (face.enrol_label === 'recommended') {
        return {
            text: `✓ Recommended +${face.coverage}`, cls: 'bg-green-600',
            title: `Clearly ${who}, a good picture, and like ${face.coverage} other tracks' unrecognised faces: adding it should help where recognition now fails`,
        };
    }
    if (face.enrol_label === 'risky') {
        const second = scores.length > 1 ? scores[1].person_name : 'another person';
        return {
            text: '⚠ Risky', cls: 'bg-orange-600',
            title: `${who} and ${second} are close: in the gallery this face may pull ${second}'s faces`,
        };
    }
    const why = face.enrol_reason === 'low_quality'
        ? `Would help (like ${face.coverage} other unrecognised faces), but the picture is too poor for the gallery`
        : 'Little expected gain: few other unrecognised faces look like it';
    return {
        text: face.coverage ? `Review +${face.coverage}` : 'Review', cls: 'bg-gray-600 opacity-75', title: why,
    };
}
