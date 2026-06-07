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
