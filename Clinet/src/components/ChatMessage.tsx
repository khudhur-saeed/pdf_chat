import { Avatar, AvatarFallback } from "./ui/avatar";
import { User, FileText } from "lucide-react";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
}

// Function to detect RTL text (Arabic, Hebrew, Persian, etc.)
function detectRTL(text: string): boolean {
  const rtlChars = /[\u0591-\u07FF\u200F\u202B\u202E\uFB1D-\uFDFD\uFE70-\uFEFC]/;
  return rtlChars.test(text);
}

// Function to parse and render markdown-like formatting
function parseMessage(text: string) {
  const parts: (string | JSX.Element)[] = [];
  let lastIndex = 0;

  // Replace **text** with bold
  const boldRegex = /\*\*(.*?)\*\*/g;
  let match;
  const boldMatches: Array<{ index: number; length: number; text: string }> = [];
  
  while ((match = boldRegex.exec(text)) !== null) {
    boldMatches.push({
      index: match.index,
      length: match[0].length,
      text: match[1]
    });
  }

  if (boldMatches.length === 0) {
    return text;
  }

  boldMatches.forEach((bold) => {
    if (bold.index > lastIndex) {
      parts.push(text.substring(lastIndex, bold.index));
    }
    parts.push(
      <strong key={`bold-${bold.index}`} className="font-bold text-foreground">
        {bold.text}
      </strong>
    );
    lastIndex = bold.index + bold.length;
  });

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts;
}

// Function to render message with line breaks and formatting
function renderMessageContent(message: string) {
  const lines = message.split('\n');
  
  return lines.map((line, idx) => {
    // Check if line is a numbered list item
    const numberedListMatch = line.match(/^\d+\.\s+/);
    // Check if line is a bullet list item
    const bulletListMatch = line.match(/^\s*[*-]\s+/);

    if (numberedListMatch || bulletListMatch) {
      return (
        <div key={idx} className="ml-4 my-1">
          {parseMessage(line)}
        </div>
      );
    }

    if (line.trim() === '') {
      return <div key={idx} className="h-2" />;
    }

    return (
      <div key={idx} className="my-1">
        {parseMessage(line)}
      </div>
    );
  });
}

export function ChatMessage({ message, isUser, timestamp }: ChatMessageProps) {
  const isRTL = detectRTL(message);
  
  return (
    <div className={`px-3 sm:px-4 md:px-6 py-4 md:py-6 ${isUser ? "bg-background" : "bg-muted/30"}`}>
      <div className="flex gap-3 md:gap-4 max-w-none">
        <Avatar className="h-7 w-7 sm:h-8 sm:w-8 flex-shrink-0">
          <AvatarFallback className={isUser ? "bg-blue-500 text-white" : "bg-orange-500 text-white"}>
            {isUser ? <User className="h-3 w-3 sm:h-4 sm:w-4" /> : <FileText className="h-3 w-3 sm:h-4 sm:w-4" />}
          </AvatarFallback>
        </Avatar>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 md:mb-2">
            <span className="text-xs sm:text-sm font-medium text-foreground">
              {isUser ? "You" : "PDF Assistant"}
            </span>
            <span className="text-xs text-muted-foreground">
              {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
          <div 
            className="text-sm sm:text-base text-foreground break-words leading-relaxed"
            dir={isRTL ? "rtl" : "ltr"}
            style={{ textAlign: isRTL ? "right" : "left" }}
          >
            {renderMessageContent(message)}
          </div>
        </div>
      </div>
    </div>
  );
}