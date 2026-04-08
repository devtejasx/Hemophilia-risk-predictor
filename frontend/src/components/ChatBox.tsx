import React from 'react'
import { Send, Loader } from 'lucide-react'
import clsx from 'clsx'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

interface ChatBoxProps {
  messages: ChatMessage[]
  onSendMessage: (message: string) => void
  loading?: boolean
  placeholder?: string
}

export const ChatBox: React.FC<ChatBoxProps> = ({
  messages,
  onSendMessage,
  loading = false,
  placeholder = 'Type your question...',
}) => {
  const [inputValue, setInputValue] = React.useState('')
  const messagesEndRef = React.useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  React.useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue)
      setInputValue('')
    }
  }

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-slate-500 dark:text-slate-400">
            <p className="text-center">
              <span className="block text-lg font-medium mb-2">No messages yet</span>
              <span className="text-sm">Start a conversation to get clinical insights</span>
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={clsx('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}
          >
            <div
              className={clsx(
                'max-w-xs lg:max-w-md px-4 py-3 rounded-lg',
                msg.role === 'user'
                  ? 'bg-purple-500 text-white rounded-br-none'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white rounded-bl-none'
              )}
            >
              <p className="text-sm">{msg.content}</p>
              {msg.timestamp && (
                <p
                  className={clsx(
                    'text-xs mt-1',
                    msg.role === 'user'
                      ? 'text-purple-100'
                      : 'text-slate-500 dark:text-slate-400'
                  )}
                >
                  {msg.timestamp}
                </p>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-100 dark:bg-slate-800 px-4 py-3 rounded-lg rounded-bl-none">
              <Loader className="w-5 h-5 animate-spin text-purple-500" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-200 dark:border-slate-800 p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder={placeholder}
            disabled={loading}
            className="flex-1 px-4 py-2 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={loading || !inputValue.trim()}
            className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatBox
