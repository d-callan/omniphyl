#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "==> Building HyPhyStudio in release configuration..."
swift build -c release

APP_NAME="HyPhy Studio"
APP_BUNDLE="$DIR/build/$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

echo "==> Constructing $APP_NAME.app bundle structure..."
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS"
mkdir -p "$RESOURCES"

# Copy executable
cp ".build/release/HyPhyStudio" "$MACOS/HyPhyStudio"

# Copy resource bundle
if [ -d ".build/release/HyPhyStudio_HyPhyStudio.bundle" ]; then
    cp -R ".build/release/HyPhyStudio_HyPhyStudio.bundle" "$RESOURCES/"
    # Also copy resource contents directly into Resources so standard Bundle lookups find them
    cp -R ".build/release/HyPhyStudio_HyPhyStudio.bundle/"* "$RESOURCES/"
fi

# Copy AppIcon if available
if [ -f "Sources/HyPhyStudio/Resources/AppIcon.icns" ]; then
    cp "Sources/HyPhyStudio/Resources/AppIcon.icns" "$RESOURCES/AppIcon.icns"
fi

# Write Info.plist
cat << 'EOF' > "$CONTENTS/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>HyPhyStudio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIconName</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>org.veg.HyPhyStudio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>HyPhy Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>3.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>HyPhy Studio Project</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>LSHandlerRank</key>
            <string>Owner</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>org.veg.hyphyproj</string>
            </array>
            <key>LSTypeIsPackage</key>
            <true/>
        </dict>
        <dict>
            <key>CFBundleTypeName</key>
            <string>NEXUS Alignment</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>org.veg.nexus</string>
                <string>public.nexus</string>
            </array>
        </dict>
        <dict>
            <key>CFBundleTypeName</key>
            <string>FASTA Alignment</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>org.veg.fasta</string>
                <string>public.fasta</string>
            </array>
        </dict>
    </array>
    <key>UTExportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key>
            <string>org.veg.hyphyproj</string>
            <key>UTTypeDescription</key>
            <string>HyPhy Studio Project Bundle</string>
            <key>UTTypeConformsTo</key>
            <array>
                <string>com.apple.package</string>
                <string>public.composite-content</string>
            </array>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array>
                    <string>hyphyproj</string>
                </array>
            </dict>
        </dict>
    </array>
    <key>UTImportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeIdentifier</key>
            <string>org.veg.nexus</string>
            <key>UTTypeDescription</key>
            <string>NEXUS Alignment File</string>
            <key>UTTypeConformsTo</key>
            <array>
                <string>public.plain-text</string>
            </array>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array>
                    <string>nex</string>
                    <string>nexus</string>
                </array>
            </dict>
        </dict>
        <dict>
            <key>UTTypeIdentifier</key>
            <string>org.veg.fasta</string>
            <key>UTTypeDescription</key>
            <string>FASTA Alignment File</string>
            <key>UTTypeConformsTo</key>
            <array>
                <string>public.plain-text</string>
            </array>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array>
                    <string>fas</string>
                    <string>fasta</string>
                    <string>fna</string>
                    <string>fa</string>
                </array>
            </dict>
        </dict>
    </array>
</dict>
</plist>
EOF

echo "==> Ad-hoc code signing $APP_NAME.app..."
/usr/bin/codesign --force --deep --sign - "$APP_BUNDLE"

echo "==> Successfully created $APP_BUNDLE"
ls -la "$MACOS"
