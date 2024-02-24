import smtplib
import yaml
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_verification_mail(email, verification_code):
    try:

        config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        print('config: ', config)
        smtp = config['smtp']
  
        msg = MIMEMultipart()
        msg['Subject'] = 'ACDC SIGN UP'
        msg['From'] = smtp['username']
        msg['To'] = email

        html_content = """
        <html>
        <head></head>
        <body>
        <p>Hello {},</p>
        <p>Thank you for signing up for the <span style="font-weight: bold; color: #1a73e8;">ACDC</span> project. Please click on the link below to verify your email address.</p>
        <p style="margin-left: 20px;"><a href="{}/api/verify?code={}&username={}">{}/api/verify?code={}&username={}</a></p>
        </body>
        </html>
        """.format(email, config['hostname'], verification_code, email, config['hostname'], verification_code, email)

        msg.attach(MIMEText(html_content, 'html'))

        smtp_server = smtp['server']
        smtp_port = smtp['port']

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(smtp['username'], smtp['password'])

        server.send_message(msg)
        server.close()

    except Exception as e:
        print("Error: ", str(e))

send_verification_mail('abdullahadeel7776@gmail.com', '123456')