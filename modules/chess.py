import chess
import json


class Chess:
    def __init__(self, width, fen, orientation="white", highlight_move=None, move_history=None, theme="classic", arrow_move=None):
        self.width = width
        self.fen = fen
        self.orientation = orientation
        self.highlight_move = highlight_move
        self.move_history = move_history if move_history else []
        self.theme = theme
        self.arrow_move = arrow_move

    def __theme_css__(self):
        themes = {
            "classic": ("#f0d9b5", "#b58863", "#5a3e1e"),
            "green":   ("#eeeed2", "#769656", "#2e4a1b"),
            "blue":    ("#dee3e6", "#8ca2ad", "#1a3a4a"),
            "gray":    ("#e8e8e8", "#7a7a7a", "#333333"),
            "purple":  ("#e8dfee", "#8878a8", "#3d2e50"),
        }
        light, dark, text = themes.get(self.theme, themes["classic"])
        return f"""
        <style>
            .white-1e1d7 {{ background-color: {light} !important; color: {text} !important; }}
            .black-3c85d {{ background-color: {dark} !important; color: {text} !important; }}
            .board-b72b1 {{ border-color: {dark} !important; }}
        </style>
        """
        self.arrow_move = arrow_move

    def __header__(self):
        return """
        <link rel="stylesheet"
            href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"
            integrity="sha384-q94+BZtLrkL1/ohfjR8c6L+A6qzNH9R2hBLwyoAfu3i/WCvQjzL2RQJ3uNHDISdU"
            crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
        <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"
            integrity="sha384-8Vi8VHwn3vjQ9eUHUxex3JSN/NFqUg3QbPyX8kWyb93+8AC/pPWTzj+nHtbC5bxD"
            crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.2/chess.js"
            integrity="sha384-s3XgLpvmHyscVpijnseAmye819Ee3yaGa8NxstkJVyA6nuDFjt59u1QvuEl/mecz"
        crossorigin="anonymous"></script>
        """

    def __header_with_theme__(self):
        return self.__header__() + self.__theme_css__()

    def __sidetomove__(self):
        return self.orientation

    def __highlight_script__(self):
        if not self.highlight_move or len(str(self.highlight_move)) < 4:
            return "// no highlight"
        hm = str(self.highlight_move)
        frm = hm[0:2]
        to = hm[2:4]
        return f"""
        setTimeout(function() {{
            var fromEl = document.querySelector('#myBoard [data-square="{frm}"]');
            var toEl = document.querySelector('#myBoard [data-square="{to}"]');
            var style = 'inset 0 0 0 4px rgba(247, 183, 51, 0.85)';
            if (fromEl) fromEl.style.boxShadow = style;
            if (toEl) toEl.style.boxShadow = style;
        }}, 300);
        """
    def __sound_script__(self):
        return """
        var soundEnabled = true;

        // Simple beep sounds using Web Audio API (no external files)
        function playSound(type) {
            if (!soundEnabled) return;
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);

                var freq = 440;
                var duration = 0.08;

                if (type === 'move') {
                    freq = 440; duration = 0.08;
                } else if (type === 'capture') {
                    freq = 300; duration = 0.12;
                } else if (type === 'check') {
                    freq = 660; duration = 0.15;
                } else if (type === 'castle') {
                    freq = 520; duration = 0.1;
                } else if (type === 'undo') {
                    freq = 220; duration = 0.15;
                } else if (type === 'gameover') {
                    freq = 180; duration = 0.4;
                }

                osc.frequency.value = freq;
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.15, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + duration);
            } catch (e) {
                // silently fail
            }
        }
        """
    
    def __sync_script__(self):
        return """
        (function() {
            var btn = document.getElementById('syncBtn');
            if (!btn) return;
            btn.onclick = function() {
                var payload = {
                    'move': null,
                    'fen': game.fen(),
                    'pgn': game.pgn(),
                    'history': moveHistory
                };
                if (window.parent) {
                    window.parent.stBridges.send('my-bridge', payload);
                } else {
                    window.stBridges.send('my-bridge', payload);
                }
                var s = document.getElementById('syncStatus');
                if (s) s.textContent = 'Sent ' + moveHistory.length + ' moves. Waiting for app...';
            };
        })();
        """
    def __arrow_script__(self):
        if not self.arrow_move or len(str(self.arrow_move)) < 4:
            return "// no arrow"
        uci = str(self.arrow_move)
        frm = uci[0:2]
        to = uci[2:4]
        w = self.width
        return f"""
        setTimeout(function() {{
            var svg = document.getElementById('arrowOverlay');
            if (!svg) {{ console.log('arrow: no svg'); return; }}
            var frm = "{frm}";
            var to = "{to}";
            var orientation = "{self.orientation}";
            var squareSize = {w} / 8;

            function sqToXY(sq) {{
                var f = sq.charCodeAt(0) - 97;
                var r = parseInt(sq[1]) - 1;
                var x, y;
                if (orientation === 'white') {{ x = f; y = 7 - r; }}
                else {{ x = 7 - f; y = r; }}
                return [ (x + 0.5) * squareSize, (y + 0.5) * squareSize ];
            }}

            var a = sqToXY(frm);
            var b = sqToXY(to);

            while (svg.firstChild) svg.removeChild(svg.firstChild);

            var ns = 'http://www.w3.org/2000/svg';

            var line = document.createElementNS(ns, 'line');
            line.setAttribute('x1', a[0]);
            line.setAttribute('y1', a[1]);
            line.setAttribute('x2', b[0]);
            line.setAttribute('y2', b[1]);
            line.setAttribute('stroke', '#e67e22');
            line.setAttribute('stroke-width', '10');
            line.setAttribute('stroke-linecap', 'round');
            line.setAttribute('opacity', '0.9');
            svg.appendChild(line);

            var angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
            var headLen = 18;
            var p1x = b[0] - headLen * Math.cos(angle - Math.PI/6);
            var p1y = b[1] - headLen * Math.sin(angle - Math.PI/6);
            var p2x = b[0] - headLen * Math.cos(angle + Math.PI/6);
            var p2y = b[1] - headLen * Math.sin(angle + Math.PI/6);
            var poly = document.createElementNS(ns, 'polygon');
            poly.setAttribute('points', b[0] + ',' + b[1] + ' ' + p1x + ',' + p1y + ' ' + p2x + ',' + p2y);
            poly.setAttribute('fill', '#e67e22');
            svg.appendChild(poly);

            console.log('arrow drawn: ' + frm + ' -> ' + to);
        }}, 500);
        """

    def __undo_script__(self):
        return """
        (function() {
            var START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            var btn = document.getElementById('undoBtn');
            if (!btn) return;
            btn.onclick = function() {
                if (moveHistory.length === 0) return;
                moveHistory.pop();
                var targetFen = moveHistory.length > 0
                    ? moveHistory[moveHistory.length - 1].fen
                    : START_FEN;
                game = new Chess(targetFen);
                try { playSound('undo'); } catch (e) {}
                board.position(targetFen, false);
                updateStatus();
                var s = document.getElementById('syncStatus');
                if (s) s.textContent = 'Moves this session: ' + moveHistory.length;
                if (sendTimer) clearTimeout(sendTimer);
                sendTimer = setTimeout(sendHistoryToApp, 300);
            };
        })();
        """
        def __undo_script__(self):
            return """
        (function() {
            var START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            var btn = document.getElementById('undoBtn');
            if (!btn) return;
            btn.onclick = function() {
                if (moveHistory.length === 0) return;
                moveHistory.pop();
                var targetFen = moveHistory.length > 0
                    ? moveHistory[moveHistory.length - 1].fen
                    : START_FEN;
                game = new Chess(targetFen);
                board.position(targetFen, false);
                updateStatus();
                var s = document.getElementById('syncStatus');
                if (s) s.textContent = 'Moves this session: ' + moveHistory.length;
                if (sendTimer) clearTimeout(sendTimer);
                sendTimer = setTimeout(sendHistoryToApp, 300);
            };
        })();
        """

    def __board_placeholder__(self):
        return f"""
        <div id="boardWrapper" style="position: relative; display: inline-block;">
            <div id="myBoard" style="width: {self.width}px"></div>
            <svg id="arrowOverlay" width="{self.width}" height="{self.width}"
                 style="position: absolute; top: 0; left: 0; pointer-events: none; z-index: 100;"></svg>
        </div>
        <br>
        <label><strong>Status:</strong></label>
        <div id="status"></div>
        <div id="syncStatus" style="color: #e67e22; font-weight: bold; margin-top: 8px;">Moves this session: 0</div>
        <button id="undoBtn" style="margin-top: 6px; padding: 8px 16px; background: #333; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">↩️ Undo Last Move</button>
        """

    def puzzle_board(self):
        _hist_json = json.dumps(self.move_history)
        script1 = f"""
        var board = null
        var game = new Chess('{self.fen}')
        var $status = $('#status')
        var moveHistory = {_hist_json};
        """

        game_over_ = """
        if (game.game_over()) return false
        if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
            (game.turn() === 'b' && piece.search(/^w/) !== -1)) return false
        """

        script2 = f"""
        function onDragStart (source, piece, position, orientation) {{{game_over_}}}

        // ---- Legal move hints ----
        function clearHints() {{
            var hints = document.querySelectorAll('.move-hint');
            hints.forEach(function(el) {{ el.remove(); }});
        }}

        function showHints(square) {{
            clearHints();
            var moves = game.moves({{square: square, verbose: true}});
            var boardEl = document.getElementById('myBoard');
            if (!boardEl) return;
            var boardRect = boardEl.getBoundingClientRect();
            var sqSize = boardRect.width / 8;
            var orientation = '{self.orientation}';

            moves.forEach(function(mv) {{
                var targetFile = mv.to.charCodeAt(0) - 97;
                var targetRank = parseInt(mv.to[1]) - 1;
                var x, y;
                if (orientation === 'white') {{
                    x = targetFile;
                    y = 7 - targetRank;
                }} else {{
                    x = 7 - targetFile;
                    y = targetRank;
                }}
                var hint = document.createElement('div');
                hint.className = 'move-hint';
                var isCapture = mv.flags.indexOf('c') !== -1 || mv.flags.indexOf('e') !== -1;
                var dotSize = isCapture ? sqSize : sqSize * 0.32;
                hint.style.position = 'absolute';
                hint.style.left = (x * sqSize + (sqSize - dotSize) / 2) + 'px';
                hint.style.top = (y * sqSize + (sqSize - dotSize) / 2) + 'px';
                hint.style.width = dotSize + 'px';
                hint.style.height = dotSize + 'px';
                hint.style.borderRadius = '50%';
                hint.style.pointerEvents = 'none';
                hint.style.zIndex = '5';
                if (isCapture) {{
                    hint.style.background = 'transparent';
                    hint.style.border = '4px solid rgba(230, 126, 34, 0.75)';
                }} else {{
                    hint.style.background = 'rgba(230, 126, 34, 0.55)';
                }}
                boardEl.parentElement.appendChild(hint);
            }});
        }}

        // Hook into Chessboard's drag
        var origOnDragStart = onDragStart;
        onDragStart = function(source, piece, position, orientation) {{
            var result = origOnDragStart(source, piece, position, orientation);
            if (result !== false) {{
                setTimeout(function() {{ showHints(source); }}, 10);
            }}
            return result;
        }};
        """

        script3 = """
        var sendTimer = null;
        function sendHistoryToApp() {
          var payload = {
            'move': null,
            'fen': game.fen(),
            'pgn': game.pgn(),
            'history': moveHistory
          };
          if (window.parent) {
            window.parent.stBridges.send('my-bridge', payload);
          } else {
            window.stBridges.send('my-bridge', payload);
          }
        }
        function onDrop (source, target) {
          var move = game.move({
            from: source,
            to: target,
            promotion: 'q'
          })
          if (move === null) return 'snapback'
          moveHistory.push({san: move.san, from: move.from, to: move.to, color: move.color, fen: game.fen()});
          updateStatus();
          // Play a sound based on the move type
          try {
            if (game.in_checkmate() || game.in_draw()) {
              playSound('gameover');
            } else if (game.in_check()) {
              playSound('check');
            } else if (move.flags.indexOf('c') !== -1 || move.flags.indexOf('e') !== -1) {
              playSound('capture');
            } else if (move.flags.indexOf('k') !== -1 || move.flags.indexOf('q') !== -1) {
              playSound('castle');
            } else {
              playSound('move');
            }
          } catch (e) {}
          var s = document.getElementById('syncStatus');
          if (s) s.textContent = 'Moves this session: ' + moveHistory.length + ' (syncing...)';
          if (sendTimer) clearTimeout(sendTimer);
          sendTimer = setTimeout(sendHistoryToApp, 500);
        }
        """

        script4 = """
        function onSnapEnd () {
          board.position(game.fen())
          var hints = document.querySelectorAll('.move-hint');
          hints.forEach(function(el) { el.remove(); });
        }

        function updateStatus () {
          var status = ''
          var moveColor = 'White'
          if (game.turn() === 'b') { moveColor = 'Black' }
          if (game.in_checkmate()) {
            status = 'Game over, ' + moveColor + ' is in checkmate.'
          } else if (game.in_draw()) {
            status = 'Game over, drawn position'
          } else {
            status = moveColor + ' to move'
            if (game.in_check()) {
              status += ', ' + moveColor + ' is in check'
            }
          }
          $status.html(status)
        }
        """

        config_ = f"""
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png',
        position: '{self.fen}',
        orientation: '{self.orientation}',
        draggable: true,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
        """

        script5 = f"""
        var config = {{{config_}}}
        board = Chessboard('myBoard', config)
        updateStatus()
        """

        ret = []
        ret.append(self.__header__())
        ret.append(self.__theme_css__())
        ret.append(self.__board_placeholder__())
        ret.append('<script>')
        ret.append(script1)
        ret.append(script2)
        ret.append(script3)
        ret.append(script4)
        ret.append(script5)
        ret.append(self.__highlight_script__())
        ret.append(self.__sync_script__())
        ret.append(self.__undo_script__())
        ret.append(self.__arrow_script__())
        ret.append(self.__sound_script__())
        ret.append('</script>')
        return '\n'.join(ret)